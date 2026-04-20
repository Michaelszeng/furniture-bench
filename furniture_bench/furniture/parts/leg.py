import random

import numpy as np
import numpy.typing as npt
import torch
from rich import print

import furniture_bench.controllers.control_utils as C
import furniture_bench.utils.transform as T
from furniture_bench.config import config
from furniture_bench.furniture.parts.part import Part
from furniture_bench.utils.pose import get_mat, is_similar_rot, is_similar_xz, rot_mat


class Leg(Part):
    # ── Non-Markovian feature toggles (active only when --non-markovian is set) ──
    _NM_PAUSE_EXCLUDED_STATES: frozenset = frozenset({"release", "lift_up", "reach_leg_floor_xy", "reach_leg_floor_z"})

    # ── Per-state noise tiers ──────────────────────────────────────────────────
    # Target-noise tiers (used by apply_non_markovian_config for latent offsets)
    _LOW_LATENT_TARGET_STD_STATES: frozenset = frozenset(
        {"reach_leg_floor_z", "pick_leg", "reach_table_top_xy", "pre_screw"}
    )
    _ZERO_LATENT_TARGET_STD_STATES: frozenset = frozenset(
        {"screw_grasp", "screw", "insert_release", "reach_table_top_z"}
    )
    # Action-noise tiers (used by furniture_sim_env when computing the executed action).
    _LOW_ACTION_NOISE_STATES: frozenset = frozenset({"reach_table_top_z", "pre_screw", "screw_grasp", "screw"})
    _NO_ACTION_NOISE_STATES: frozenset = frozenset({"insert_release"})
    # States where the noisy (executed) action is also recorded as the clean action.
    _CLEAN_ACTION_NOISE_STATES: frozenset = frozenset({"reach_table_top_z"})

    # Ry angle (radians) that pitches the EE toward the floor during the floor pick-up.
    # Adjust here to change the grasp tilt; used identically in compute_state and fsm_step.
    _GRASP_MARGIN_ANGLE: float = -np.pi / 7
    _INSERT_TIP_RY_ANGLE: float = np.radians(4.5)  # slight y-axis tilt during table-top approach and insertion
    _LEG_TIP_OFFSET: float = 0.05625  # distance from leg mesh origin to screw tip (m)
    _LEG_HOLE_OFFSET_X: float = 0.0025  # fine-alignment offset of tip to table hole, X (m)
    _LEG_HOLE_OFFSET_Y: float = 0.001  # fine-alignment offset of tip to table hole, Y (m)
    _PICK_X_OFFSET: float = 0.005  # grab 0.5 cm toward the "top" of the leg along X
    _PICK_Z_OFFSET: float = 0.012  # EE hovers 1.2 cm above the leg COM during floor pick
    _ORI_Z_CLEARANCE: float = 0.05  # extra Z clearance during orientation alignment

    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        tag_ids = part_config["ids"]

        self.rel_pose_from_center[tag_ids[0]] = get_mat([0, 0, -self.tag_offset], [0, 0, 0])
        self.rel_pose_from_center[tag_ids[1]] = get_mat([-self.tag_offset, 0, 0], [0, np.pi / 2, 0])
        self.rel_pose_from_center[tag_ids[2]] = get_mat([0, 0, self.tag_offset], [0, np.pi, 0])
        self.rel_pose_from_center[tag_ids[3]] = get_mat([self.tag_offset, 0, 0], [0, -np.pi / 2, 0])

        self.done = False
        self.pos_error_threshold = 0.01
        self.ori_error_threshold = 0.25

        self.skill_complete_next_states = [
            "lift_up",
            "pre_screw",
        ]  # Specify next state after a skill is complete. Screw done is handled in `get_assembly_action`

        self.reset()

        self.part_attached_skill_idx = 4

    def reset(self):
        super().reset()  # resets prev_cnt=0, curr_cnt=0, pre_assemble_done, latent_offsets, step_noise_*, etc.
        self._last_state = "reach_leg_floor_xy"
        self.GOT_STUCK = False
        self.prev_leg_tip_z_rel = float("inf")
        self.prev_leg_z_vel_robot = float("inf")
        self.gripper_action = -1

    def apply_non_markovian_config(self):
        """Sample episode-level non-Markovian latent variables."""
        if not self._NM_LATENT_PLAN:
            return

        HIGH_STD = 0.020  # m — persistent position offset for coarse-motion states
        LOW_STD = 0.003  # m — persistent position offset for precision states
        HIGH_ORI_STD = np.radians(5.0)  # rad — persistent orientation offset for coarse-motion states
        LOW_ORI_STD = np.radians(2.0)  # rad — persistent orientation offset for precision states

        all_states = [
            "reach_leg_floor_xy",
            "reach_leg_ori",
            "reach_leg_floor_z",
            "pick_leg",
            "lift_up",
            "match_leg_ori",
            "reach_table_top_xy",
            "back_out_for_retry",
            "reach_table_top_z",
            "insert_release",
            "release",
            "pre_screw",
            "screw_grasp",
            "screw",
        ]

        def pos_std_for(state):
            if state in self._ZERO_LATENT_TARGET_STD_STATES:
                return 0.0
            if state in self._LOW_LATENT_TARGET_STD_STATES:
                return LOW_STD
            return HIGH_STD

        def ori_std_for(state):
            if state in self._ZERO_LATENT_TARGET_STD_STATES:
                return 0.0
            if state in self._LOW_LATENT_TARGET_STD_STATES:
                return LOW_ORI_STD
            return HIGH_ORI_STD

        self.latent_offsets = {
            state: {
                "pos": np.random.normal(0, pos_std_for(state), size=(3,)),
                "ori": np.random.normal(0, ori_std_for(state), size=(3,)),
            }
            for state in all_states
        }

    def is_in_reset_ori(self, pose: npt.NDArray[np.float32], from_skill, ori_bound) -> bool:
        # y-axis of the leg align with y-axis of the base.
        reset_ori = self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        for _ in range(4):
            if is_similar_rot(pose[:3, :3], reset_ori[:3, :3], ori_bound=ori_bound):
                return True
            pose = pose @ rot_mat(np.array([0, np.pi / 2, 0]), hom=True)
        return False

    def _find_down_z(self, mat):
        for _ in range(4):
            if mat[2, 2] > 0.8:  # Z is down.
                break
            mat = mat @ torch.tensor(T.rotmat2hom(rot_mat([0, np.pi / 2, 0]))).float().to(mat.device)
        return mat

    def compute_state(
        self,
        ee_pos,
        ee_quat,
        gripper_width,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        assemble_to,
        ee_ang_vel=None,
    ) -> str:
        """Determine the FSM state from current environment state (hierarchical).

        Three mutually exclusive top-level phases:
          GRASPED  — gripper is in contact with leg (gripper_width < 2*half_width + 0.005)
          INSERTED — leg is near the table hole but gripper is open
          ON_FLOOR — leg is away from hole and gripper is open
        """
        device = ee_pos.device

        ################################################################################################################
        # GET CURRENT ROBOT AND ENVIRONMENT STATE
        ################################################################################################################
        # EE Pose in robot base frame
        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))

        # Table pose in robot base frame
        table_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )  # Table pose in world frame
        table_pose = sim_to_april_mat @ table_pose  # Table pose in april frame
        table_pose_robot = april_to_robot @ table_pose  # Table pose in robot base frame

        # Leg pose in robot base frame
        leg_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )  # Leg pose in world frame
        leg_pose = sim_to_april_mat @ leg_pose  # Leg pose in april frame
        leg_pose_robot = april_to_robot @ leg_pose  # Leg pose in robot base frame

        # Table hole position in robot base frame
        table_hole_pose_robot = (
            april_to_robot
            @ table_pose
            @ torch.tensor(
                get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                device=device,
            ).float()
        )  # Table hole pose in robot base frame
        table_hole_pos_robot = table_hole_pose_robot[:3, 3]  # Table hole position in robot base frame

        # Leg z-velocity in robot frame
        _leg_vel_sim = rb_states[part_idxs[self.name]][0][7:10]
        _rot_sim_to_robot = (april_to_robot @ sim_to_april_mat)[:3, :3]
        leg_z_vel_robot = (_rot_sim_to_robot @ _leg_vel_sim)[2]

        # Leg screw tip height above table surface
        leg_z_rel = leg_pose_robot[2, 3] - table_pose_robot[2, 3]
        leg_tip_z_rel = leg_z_rel - leg_pose_robot[2, 1] * self._LEG_TIP_OFFSET

        ################################################################################################################
        # SHARED TARGET POSES
        ################################################################################################################
        ### DEFINE TARGET POSES ###
        leg_xy = leg_pose_robot[:2, 3]
        leg_z = leg_pose_robot[2, 3]
        # Offset pick target: _PICK_X_OFFSET cm toward the top of the leg in X, _PICK_Z_OFFSET cm above leg COM in Z.
        pick_target_xy = leg_xy.clone()
        pick_target_xy[0] += self._PICK_X_OFFSET
        pick_target_z = leg_z + self._PICK_Z_OFFSET

        # Top-level phase discriminators
        # Use leg tip (screw threads) rather than COM for the XY proximity check.
        # Tip is LEG_TIP_OFFSET m in the -local-Y direction from the mesh origin.
        _leg_tip_xy = leg_pose_robot[:2, 3] - leg_pose_robot[:2, 1] * self._LEG_TIP_OFFSET
        leg_xy_near_hole = torch.norm(_leg_tip_xy - table_hole_pos_robot[:2]) < 0.009
        leg_xy_near_hole_loose = torch.norm(_leg_tip_xy - table_hole_pos_robot[:2]) < 0.015

        # gripper_grasped: leg is physically between the fingers.
        # When the leg (diameter = 2*half_width) is held, physics prevents the gripper
        # from closing below 2*half_width, so the reading stays near that value.
        # Two-sided bound:
        #   upper: gripper hasn't opened past the leg (not released)
        #   lower: gripper isn't empty-closed (width ≈ 0 when nothing is held)
        # A 0.005 m upper margin and 0.010 m lower margin accommodate measurement noise.
        _leg_diameter = 2 * self.half_width
        gripper_grasped = (
            gripper_width < _leg_diameter + 0.0055  # not open
            and gripper_width > _leg_diameter - 0.010  # not empty-closed
        )

        # Precompute orientations
        _margin = C.rot_mat_tensor(0, self._GRASP_MARGIN_ANGLE, 0, device)
        # grasp_ori: base floor-pickup orientation rotated around world-Z to match the
        # leg's current yaw (same logic as floor_grasp_ori() in fsm_step).
        #
        # After _find_down_z, col-0 of leg_pose_down_robot is the leg's long axis direction.
        # At the default pose (reset_ori = Ry(-π/2)), one local Ry(π/2) gives approximately
        # identity, so leg_pose_down_robot[:3,:3] ≈ april_to_robot[:3,:3]. The default
        # long-axis yaw is therefore atan2(april_to_robot[1,0], april_to_robot[0,0]).
        _leg_pose_down = self._find_down_z(leg_pose)
        _leg_pose_down_robot = april_to_robot @ _leg_pose_down
        _leg_long_yaw = torch.atan2(_leg_pose_down_robot[1, 0], _leg_pose_down_robot[0, 0])
        _default_long_yaw = torch.atan2(april_to_robot[1, 0], april_to_robot[0, 0])
        _theta = (_leg_long_yaw - _default_long_yaw).item()
        # Wrap to [-π/2, π/2]: 180° symmetry (leg can be grasped from either perpendicular side).
        _theta_wrapped = ((float(_theta) + np.pi / 2) % np.pi) - np.pi / 2
        _grasp_base = (_margin @ april_to_robot @ C.rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device))[:3, :3]
        _Rz = torch.tensor(
            [
                [np.cos(_theta_wrapped), -np.sin(_theta_wrapped), 0.0],
                [np.sin(_theta_wrapped), np.cos(_theta_wrapped), 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        grasp_target_ori = _Rz @ _grasp_base
        insert_target_ori = (_margin @ C.rot_mat_tensor(np.pi, 0, 0, device))[
            :3, :3
        ]  # Ry(-36°) @ Rx(180°) — matches match_leg_ori target
        _table_top_z = table_pose_robot[2, 3]
        insert_target_z = _table_top_z + 0.056

        # Hard-coded INSERTED-phase orientations.
        # pre_screw = Rx(π)@Rz(π),  screw_done = Rx(π)  (−180° around Z)
        pre_screw_target_ori = C.rot_mat_tensor(np.pi, 0, np.pi, device)[:3, :3]
        screw_target_ori = C.rot_mat_tensor(np.pi, 0, 0, device)[:3, :3]

        # Stable pre-screw position: table hole XY + insert_z height.
        # Using the stable table reference (not the bouncing leg) avoids the EE chasing
        # the leg downward immediately after insertion.
        pre_screw_target_pos = table_hole_pos_robot[:3].clone()
        pre_screw_target_pos[2] = insert_target_z

        at_pre_screw_target_pos = (ee_pos - (pre_screw_target_pos)).abs().sum() < self.pos_error_threshold * 3
        at_pre_screw_target_ori = (ee_pose[:3, :3] - pre_screw_target_ori).abs().sum() < self.ori_error_threshold * 1.5

        # Staging position during "match_leg_ori" and before "reach_table_top_xy"
        self.staging_pos = torch.tensor([0.45, 0.15, 0.14], device=device)

        ################################################################################################################
        # SHARED CONDITIONS
        # _lo() returns zeros when non_markovian=False (latent_offsets is never populated),
        # so these booleans are correct for both the NM and Markovian paths.
        ################################################################################################################
        # EE orientation checks (used across multiple GRASPED sub-phases)
        ee_at_insert_ori = (ee_pose[:3, :3] - insert_target_ori).abs().sum() < self.ori_error_threshold * 3
        # Loose bound: used to decide when the EE has sufficiently rotated to proceed toward the hole.
        ee_at_insert_ori_loose = (ee_pose[:3, :3] - insert_target_ori).abs().sum() <= self.ori_error_threshold * 4
        screw_done = (ee_pose[:3, :3] - screw_target_ori).abs().sum() < self.ori_error_threshold * 1.5

        # Transport sub-phase: leg is clear of the floor after lift_up.
        leg_lifted = leg_pose_robot[2, 3] > 0.05
        # For safety, "match_leg_ori" is only returned when EE is still near the staging position;
        # once the EE has flown far toward the hole, we should not come back to match_leg_ori.
        near_staging = (ee_pos[:2] - self.staging_pos[:2]).abs().sum() < 0.2

        # Sometimes, due to action noise, leg_xy_near_hole may become untrue momentarily.
        # If the leg is still close to the hole and the robot is moving downward, we know we're probably still in
        # "reach_table_top_z".
        barely_missed_hole = leg_xy_near_hole_loose and leg_z_vel_robot < -0.06 and ee_at_insert_ori

        # Insertion depth: leg tip Z relative to table surface (in robot frame).
        # Computed unconditionally so it can be used in both the NM and Markovian paths.
        leg_z_rel = leg_pose_robot[2, 3] - table_pose_robot[2, 3]
        leg_tip_z_rel = leg_z_rel - leg_pose_robot[2, 1] * self._LEG_TIP_OFFSET
        leg_fully_inserted_z = leg_tip_z_rel < 0.057 - self._LEG_TIP_OFFSET  # tip has cleared the hole threshold

        # ON_FLOOR floor-pick approach conditions
        at_pick_xy = (ee_pos[:2] - pick_target_xy).abs().sum() < self.pos_error_threshold * 3
        # Gate on orientation alignment between EE and leg.
        # Randomized gate to get some grasp angle variation.
        at_grasp_ori_floor = (ee_pose[:3, :3] - grasp_target_ori).abs().sum() < self.ori_error_threshold * 3
        at_pick_z = abs(ee_pos[2] - pick_target_z) < self.pos_error_threshold

        ################################################################################################################
        # NON-MARKOVIAN: sequential state machine
        ################################################################################################################
        # TODO

        ################################################################################################################
        # MARKOVIAN: FSM state determined entirely via environment state
        ################################################################################################################
        # ── Phase 1: GRASPED ─────────────────────────────────────────────────
        # Primary discriminator: leg is physically between the gripper fingers.
        if gripper_grasped:
            if leg_xy_near_hole:
                # ── Screw vs Insertion sub-phase ─────────────────────────────
                # Both "carrying the leg down for insertion" and "re-grasped for screwing"
                # share the same XY proximity to the hole.  The EE orientation is the discriminator:
                #   insert_ori (Ry(-36°) @ Rx(180°)) → still inserting
                #   anything else                     → screw sub-phase
                if ee_at_insert_ori:
                    # ── Insertion sub-phase ──────────────────────────────────
                    # EE is at insert_ori, descending to insert the leg.
                    if leg_fully_inserted_z:  # tip has cleared the hole threshold → release the leg
                        return "insert_release"

                    # Detect stuck: leg is in the insertion zone but tip XY is off-center from the hole.
                    # Tunable thresholds:
                    STUCK_Z_LOW = 0.01475  # Below this means the leg is inserted
                    STUCK_Z_HIGH = 0.01675  # upper bound of stuck zone (leg hasn't made progress)
                    STUCK_Z_VEL_THRESHOLD = 0.025  # m/s — leg moving upward indicates it is trying the retry behavior
                    # 2-markovian -- check that previous state also stuck
                    z_range_stuck = leg_tip_z_rel < STUCK_Z_HIGH and self.prev_leg_tip_z_rel < STUCK_Z_HIGH
                    self.prev_leg_tip_z_rel = leg_tip_z_rel  # This is the last time leg_tip_z_rel is used in this func.
                    z_vel_stuck = (
                        leg_z_vel_robot > STUCK_Z_VEL_THRESHOLD
                    )  # If robot is already moving upward, i.e. already backing out for retry, then continue doing so
                    print(f"leg_tip_z_rel: {leg_tip_z_rel}, leg_z_vel_robot: {leg_z_vel_robot}")
                    if (
                        leg_tip_z_rel > STUCK_Z_LOW  # Don't trigger stuck if leg is inserted
                        and leg_z_vel_robot > -0.05  # Leg is not moving downward
                        and self.prev_leg_z_vel_robot > -0.05  # Leg was not moving downward in the previous step
                        and (z_range_stuck or z_vel_stuck)
                    ):
                        self.GOT_STUCK = True
                        # return "back_out_for_retry"
                        print("GOT_STUCK -- returning to reach_table_top_xy")
                        self.prev_leg_z_vel_robot = (
                            leg_z_vel_robot  # This is the last time leg_z_vel_robot is used in this func.
                        )
                        return "reach_table_top_xy"
                    self.prev_leg_z_vel_robot = (
                        leg_z_vel_robot  # This is the last time leg_z_vel_robot is used in this func.
                    )
                    return "reach_table_top_z"  # descending to insert the leg

                else:
                    # ── Screw sub-phase ──────────────────────────────────────
                    # EE is rotating from pre_screw_ori toward screw_ori.
                    # screwing complete (EE has rotated to screw_ori) → release
                    if screw_done:
                        return "release"
                    return "screw"

            # Sometimes, due to action noise, leg_xy_near_hole may become untrue momentarily.
            # If the leg is still close to the hole and the robot is moving downward, we know we're probably still in
            # "reach_table_top_z".
            elif barely_missed_hole:
                return "reach_table_top_z"  # descending to insert the leg

            else:
                # ── Transport sub-phase ──────────────────────────────────────
                # Leg is grasped and the EE is not yet over the hole.
                # Navigate in order: lift → center → orient → fly to hole.

                # "match_leg_ori": leg is lifted (Z > 0.05 m) after "lift_up" → move to the
                # staging position and rotate to insert_ori.
                # For safety, after getting far enough away from staging_pos, we don't return to "match_leg_ori"
                if leg_lifted and not ee_at_insert_ori_loose and near_staging:
                    return "match_leg_ori"

                # "reach_table_top_xy": EE orientation already matches insert_ori
                # (Ry(-36°) @ Rx(180°)) → fly EE XY to the hole position.
                if ee_at_insert_ori_loose:
                    return "reach_table_top_xy"

                # "lift_up": default — leg just grasped and still near the
                # floor (leg Z ≤ 0.05 m) → lift until leg clears the floor.
                return "lift_up"

        # ── Phase 2: INSERTED ────────────────────────────────────────────────
        # Primary discriminator: leg XY is near the hole AND gripper is open.
        # The leg was released into the hole.
        # Screw sub-sequence: pre_screw → screw_grasp → screw.
        elif leg_xy_near_hole:
            # "screw_grasp": at pre_screw_target_ori AND at position → close gripper.
            if at_pre_screw_target_ori and at_pre_screw_target_pos:
                return "screw_grasp"

            # "pre_screw": move to table hole position and rotate to pre_screw_ori.
            # This is the only INSERTED sub-state before screw_grasp — no intermediate
            # pre_grasp step. A single state eliminates any threshold-chattering between
            # two sequential states. The OSC controller routes the EE from insert_ori
            # directly to pre_screw_ori.
            return "pre_screw"

        # ── Phase 3: ON_FLOOR ────────────────────────────────────────────────
        # Default phase: leg is far from the hole (or at its initial position)
        # and gripper is open.
        else:
            # Gate on XY proximity between EE and pick target.
            if at_pick_xy:
                # Gate on orientation alignment between EE and leg.
                # Randomized gate to get some grasp angle variation.
                if at_grasp_ori_floor:
                    # Gate on Z proximity between EE and pick target.
                    if at_pick_z:
                        return "pick_leg"  # Ready to pick up the leg.

                    return "reach_leg_floor_z"  # EE is at pick XY + grasp_ori but not yet at pick Z → descend.

                return "reach_leg_ori"  # EE is close to pick XY but not yet aligned with grasp_ori → rotate in place.

            return "reach_leg_floor_xy"

    def fsm_step(
        self,
        ee_pos,
        ee_quat,
        gripper_width,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        assemble_to,
        ee_ang_vel=None,
    ):
        def rel_rot_mat(s, t):
            s_inv = torch.linalg.inv(s)
            return t @ s_inv

        def floor_grasp_ori(leg_pose_down_april):
            """Base grasp orientation rotated around world-Z to match the leg's current yaw.

            After _find_down_z, col-0 of leg_pose_down_robot is the leg's long axis direction
            in the robot frame. At the default pose (reset_ori = Ry(-π/2)), one local Ry(π/2)
            gives approximately identity, so leg_pose_down_robot[:3,:3] ≈ april_to_robot[:3,:3].
            The default long-axis yaw is therefore atan2(april_to_robot[1,0], april_to_robot[0,0]).
            theta = current_long_yaw - default_long_yaw is wrapped to [-π/2, π/2] for 180°
            symmetry (the leg can be grasped from either perpendicular side).
            """
            leg_pose_down_robot = april_to_robot @ leg_pose_down_april
            leg_long_yaw = torch.atan2(leg_pose_down_robot[1, 0], leg_pose_down_robot[0, 0])
            default_long_yaw = torch.atan2(april_to_robot[1, 0], april_to_robot[0, 0])
            theta = (leg_long_yaw - default_long_yaw).item()
            # Wrap to [-π/2, π/2]: 180° symmetry.
            theta_wrapped = ((float(theta) + np.pi / 2) % np.pi) - np.pi / 2
            base = (margin @ april_to_robot @ C.rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device))[:3, :3]
            c_d, s_d = np.cos(theta_wrapped), np.sin(theta_wrapped)
            Rz = torch.tensor(
                [[c_d, -s_d, 0.0], [s_d, c_d, 0.0], [0.0, 0.0, 1.0]],
                device=device,
                dtype=torch.float32,
            )
            return Rz @ base

        state = self.compute_state(
            ee_pos,
            ee_quat,
            gripper_width,
            rb_states,
            part_idxs,
            sim_to_april_mat,
            april_to_robot,
            assemble_to,
            ee_ang_vel=ee_ang_vel,
        )

        # Reset timeout window as soon as a new state is entered, before action code runs.
        if state != self._last_state:
            self.prev_cnt = self.curr_cnt

        # Throttled state print (every 10 steps)
        if self.curr_cnt % 10 == 0:
            print(f"[blue][LEG][/blue] state={state} (step={self.curr_cnt})")

        timeout_failure = False

        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))

        table_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        leg_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        table_pose = sim_to_april_mat @ table_pose
        leg_pose = sim_to_april_mat @ leg_pose

        margin = C.rot_mat_tensor(0, self._GRASP_MARGIN_ANGLE, 0, ee_pose.device)
        device = ee_pose.device

        def find_leg_pose_x_look_front(leg_pose):
            best_leg_pose = leg_pose.clone()
            tmp_leg_pose = leg_pose
            rot = C.rot_mat_tensor(0, -np.pi / 2, 0, device)
            for i in range(3):
                tmp_leg_pose = tmp_leg_pose @ rot
                if best_leg_pose[0, 0] < tmp_leg_pose[0, 0]:
                    best_leg_pose = tmp_leg_pose
            return best_leg_pose

        # Default target to current EE pose (overridden in each state)
        target = ee_pose.clone()
        clean_target = ee_pose.clone()

        # Precompute for states that navigate relative to the leg tip and table hole
        # (reach_table_top_xy, reach_table_top_z, insert_release).
        # find_leg_pose_x_look_front only applies Ry rotations, so col-1 (local-Y) is preserved.
        leg_pose_robot = find_leg_pose_x_look_front(april_to_robot @ leg_pose)
        leg_tip_pose_robot = leg_pose_robot.clone()
        leg_tip_pose_robot[:3, 3] = leg_pose_robot[:3, 3] - leg_pose_robot[:3, 1] * self._LEG_TIP_OFFSET
        # Pose of the table hole in the robot frame.
        table_hole_pose_robot = (
            april_to_robot
            @ table_pose
            @ torch.tensor(
                get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                device=device,
            )
        )

        if state == "reach_leg_floor_xy":
            leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
            pos = leg_pose_down[:4, 3]
            target_pos = (april_to_robot @ pos)[:3]
            # Compute the full grasp orientation (yaw-matched to leg) to extract its
            # world-Z yaw, then blend that yaw into the current EE orientation so the
            # gripper begins rotating to the correct yaw during the XY approach.
            target_ori_full = floor_grasp_ori(leg_pose_down)
            yaw_target = torch.atan2(target_ori_full[1, 0], target_ori_full[0, 0])
            yaw_current = torch.atan2(ee_pose[1, 0], ee_pose[0, 0])
            delta_yaw = yaw_target - yaw_current
            c, s = torch.cos(delta_yaw), torch.sin(delta_yaw)
            R_z_delta = torch.eye(3, device=device, dtype=ee_pose.dtype)
            R_z_delta[0, 0] = c
            R_z_delta[0, 1] = -s
            R_z_delta[1, 0] = s
            R_z_delta[1, 1] = c
            target_ori = R_z_delta @ ee_pose[:3, :3]
            target_pos[0] + self._PICK_X_OFFSET
            target_pos[2] = ee_pos[2]
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target, pos_std=0.007, ori_std_deg=5.0, step_noise_pos_std=0.01, step_noise_ori_std_deg=5.0
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.007, ori_std_deg=5.0)
            # Large XY motion from neutral to leg (can be 10-15 cm); needs many steps.
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.02, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "reach_leg_ori":
            leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
            target_ori = floor_grasp_ori(leg_pose_down)
            leg_pos_robot = (april_to_robot @ leg_pose_down[:4, 3])[:3]
            target_pos = leg_pos_robot.clone()
            target_pos[0] = leg_pos_robot[0] + self._PICK_X_OFFSET
            target_pos[2] = leg_pos_robot[2] + self._ORI_Z_CLEARANCE  # keep clearance above leg during EE rotation
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target, pos_std=0.01, ori_std_deg=2.5, step_noise_pos_std=0.01, step_noise_ori_std_deg=2.5
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.01, ori_std_deg=5)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.015, ori_error_threshold=0.1, max_len=60)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "reach_leg_floor_z":
            leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
            target_ori = floor_grasp_ori(leg_pose_down)
            leg_pos_robot = (april_to_robot @ leg_pose_down[:4, 3])[:3]  # leg COM in the robot frame
            target_pos = leg_pos_robot.clone()
            target_pos[0] = leg_pos_robot[0] + self._PICK_X_OFFSET
            target_pos[2] = leg_pos_robot[2] + self._PICK_Z_OFFSET
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            target = self._add_noise_to_target(
                clean_target, pos_std=0.01, ori_std_deg=12.0, step_noise_pos_std=0.005, step_noise_ori_std_deg=2.0
            )
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.015, ori_error_threshold=0.3, max_len=30)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "pick_leg":
            leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
            target_ori = floor_grasp_ori(leg_pose_down)
            leg_pos_robot = (april_to_robot @ leg_pose_down[:4, 3])[:3]  # leg COM in the robot frame
            target_pos = leg_pos_robot.clone()
            target_pos[0] = leg_pos_robot[0] + self._PICK_X_OFFSET
            target_pos[2] = leg_pos_robot[2] + self._PICK_Z_OFFSET
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target, pos_std=0.008, ori_std_deg=15.0, step_noise_pos_std=0.002, step_noise_ori_std_deg=2.0
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.01, ori_std_deg=15.0)
            self.gripper_action = 1
            result = self.gripper_less(gripper_width, 2 * self.half_width + 0.001)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "lift_up":
            leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
            leg_pos_robot = (april_to_robot @ leg_pose_down[:4, 3])[:3]  # leg COM in the robot frame
            target_pos = leg_pos_robot.clone()
            target_pos[2] += 0.06
            target_ori = ee_pose[:3, :3]
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(clean_target, step_noise_pos_std=0.020, step_noise_ori_std_deg=5.0)
            else:
                target = self._add_noise_to_target(clean_target)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "match_leg_ori":
            target_ori = (margin @ C.rot_mat_tensor(np.pi, 0, 0, device))[:3, :3]
            target_pos = self.staging_pos
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target, pos_std=0.002, ori_std_deg=0.5, step_noise_pos_std=0.020, step_noise_ori_std_deg=5.0
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.002, ori_std_deg=0.5)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "reach_table_top_xy" or state == "back_out_for_retry":
            target_z = 0.14 if state == "reach_table_top_xy" else 0.1
            target_leg_tip_pose_robot = torch.tensor(
                [  # Target for leg TIP: LEG_HOLE_OFFSET_X/Y align tip with table hole
                    [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3] + self._LEG_HOLE_OFFSET_X],
                    [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3] + self._LEG_HOLE_OFFSET_Y],
                    [0.0, 1.0, 0.0, table_pose[2, 3] + target_z - self._LEG_TIP_OFFSET],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            # Rotate orientation only (not position) slightly about world y-axis during insertion approach
            _c, _s = np.cos(self._INSERT_TIP_RY_ANGLE), np.sin(self._INSERT_TIP_RY_ANGLE)
            ry3 = torch.tensor(
                [[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]],
                dtype=torch.float32,
                device=device,
            )
            target_leg_tip_pose_robot[:3, :3] = ry3 @ target_leg_tip_pose_robot[:3, :3]  # Apply y-axis rotation
            rel = rel_rot_mat(leg_tip_pose_robot, target_leg_tip_pose_robot)
            clean_target = rel @ ee_pose
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target, pos_std=0.002, ori_std_deg=0.5, step_noise_pos_std=0.010, step_noise_ori_std_deg=5.0
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.002, ori_std_deg=0.5)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.015, ori_error_threshold=0.3, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "reach_table_top_z":
            target_leg_tip_pose_robot = torch.tensor(
                [  # Target for leg TIP: LEG_HOLE_OFFSET_X/Y align tip with table hole
                    [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3] + self._LEG_HOLE_OFFSET_X],
                    [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3] + self._LEG_HOLE_OFFSET_Y],
                    [0.0, 1.0, 0.0, table_pose[2, 3] + 0.05 - self._LEG_TIP_OFFSET],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            # Rotate orientation only (not position) slightly about world y-axis during insertion approach
            _c, _s = np.cos(self._INSERT_TIP_RY_ANGLE), np.sin(self._INSERT_TIP_RY_ANGLE)
            ry3 = torch.tensor(
                [[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]],
                dtype=torch.float32,
                device=device,
            )
            target_leg_tip_pose_robot[:3, :3] = ry3 @ target_leg_tip_pose_robot[:3, :3]  # Apply y-axis rotation
            rel = rel_rot_mat(leg_tip_pose_robot, target_leg_tip_pose_robot)
            clean_target = rel @ ee_pose
            # NOTE: DO NOT _apply_latent_offset() here, since we need precision during insertion
            # Don't add as much noise if the leg got stuck
            # Note that this is non-markovian but doesn't make the policy non-markovian,
            # since this only affects the noise added during simulation, doesn't affect the actual recorded actions.
            if not self.GOT_STUCK:
                if self.non_markovian:
                    target = self._add_noise_to_target(
                        clean_target,
                        pos_std=0.002,
                        pos_std_z=0.0,
                        ori_std_deg=2.0,
                        pos_max=0.005,
                        ori_max_deg=4.0,
                        step_noise_pos_std=0.002,
                        step_noise_pos_std_z=0.0,
                        step_noise_ori_std_deg=1.0,
                    )
                else:
                    target = self._add_noise_to_target(
                        clean_target,
                        pos_std=0.004,
                        pos_std_z=0.0,
                        ori_std_deg=2.5,
                        pos_max=0.005,
                        ori_max_deg=4.0,
                        step_noise_pos_std=0.003,
                    )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.0, ori_std_deg=0.0)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.007, ori_error_threshold=0.15, max_len=75)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "insert_release":
            self.gripper_action = -1
            # Sub-phase gate: hold EE in place until the gripper has cleared the leg,
            # then move upward to the pre-screw position.
            gripper_cleared_leg = gripper_width >= 2 * self.half_width + 0.010
            if gripper_cleared_leg:
                target_leg_tip_pose_robot = torch.tensor(
                    [  # Target for leg TIP: LEG_HOLE_OFFSET_X/Y align tip with table hole
                        [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3] + self._LEG_HOLE_OFFSET_X],
                        [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3] + self._LEG_HOLE_OFFSET_Y],
                        [0.0, 1.0, 0.0, table_pose[2, 3] + 0.084 - self._LEG_TIP_OFFSET],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    device=device,
                )
                rel = rel_rot_mat(leg_tip_pose_robot, target_leg_tip_pose_robot)
                clean_target = rel @ ee_pose
                clean_target = self._apply_latent_offset(state, clean_target)
                target = self._add_noise_to_target(
                    clean_target, pos_std=0.001, ori_std_deg=1.0
                )  # zero step noise (ZERO tier)
            else:
                # Gripper still closing around the leg — hold EE position, keep opening.
                clean_target = ee_pose.clone()
                target = clean_target
            result = self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["square_table"] - 0.001,
            )
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "release":
            target_ori = C.rot_mat_tensor(np.pi, 0, 0, device)[:3, :3]
            target_pos = (april_to_robot @ leg_pose[:4, 3])[:3]
            target_pos[2] += self.grasp_margin_z
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target, pos_std=0.001, ori_std_deg=1.0, step_noise_pos_std=0.003, step_noise_ori_std_deg=1.0
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.001, ori_std_deg=1.0)
            self.gripper_action = -1
            result = self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["square_table"] - 0.001,
            )
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "pre_screw":
            target_pos = (april_to_robot @ leg_pose)[:3, 3].clone()
            target_pos[2] = 0.055 if not self._NM_STEP_NOISE else 0.061  # A little higher if there is target step noise

            ee_z_dot_down = -ee_pose[2, 2]  # 1.0 = EE Z-axis straight down
            ee_x_dot_world_x = ee_pose[0, 0]  # X-component of EE local-X (world frame)

            # 2-stage sequence: first right the gripper (so it is vertical); then, rotate +180 deg about
            # world-Z to pre_screw_ori.
            print(f"ee_z_dot_down: {ee_z_dot_down}")
            if ee_z_dot_down < 0.9975:
                # Phase 1: right the gripper vertically
                target_ori = C.rot_mat_tensor(np.pi, 0, 0, device)[:3, :3]
            else:
                # 180° rotation split into two 90° phases to avoid ambiguity.
                if ee_x_dot_world_x > 0.25:
                    # Phase 2a: first 90° CCW about world-Z.
                    print("phase 2a")
                    target_ori = C.rot_mat_tensor(np.pi, 0, np.pi / 2, device)[:3, :3]
                else:
                    # Phase 2b: final 90° to full pre_screw_ori.
                    print("phase 2b")
                    target_ori = C.rot_mat_tensor(np.pi, 0, np.pi, device)[:3, :3]

            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target, pos_std=0.0, ori_std_deg=0.0, step_noise_pos_std=0.003, step_noise_ori_std_deg=1.0
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.0, ori_std_deg=0.0)
            result = self.satisfy(ee_pose, target, max_len=300)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "screw_grasp":
            # IMPORTANT: define target relative to leg pose so that we grab it centered
            target_pos = (april_to_robot @ leg_pose)[:3, 3].clone()
            # target_pos[2] = table_pose_robot[2, 3] + 0.065
            target_pos[2] = 0.055
            target_ori = C.rot_mat_tensor(np.pi, 0, np.pi, device)[:3, :3]
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            target = self._add_noise_to_target(clean_target, pos_std=0.0, ori_std_deg=0.0)  # NO NOISE during grasp
            self.gripper_action = 1
            result = self.gripper_less(gripper_width, 2 * self.half_width + 0.001)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "screw":
            # Rotate EE -180° about global Z from pre_screw back to Rx(π).
            # Split into two 90° steps (same technique as pre_screw) to avoid rotational ambiguity.
            ee_x_dot_world_x = ee_pose[0, 0]

            # Simply hard-code target position during screwing
            # Note: intentionally make the target height during screwing 1mm above the height during screw_grasp
            # This way, the gripper doesn't apply too much force into the table_top
            # ACTUALLY: the downward pressure seems more stable, but it makes the dynamics a little more predictable.
            # So doesn't require as much reactivity.
            TARGET_EE_POS = [0.6453, 0.1375]
            target_pos = torch.tensor(
                [TARGET_EE_POS[0], TARGET_EE_POS[1], 0.056],
                # [TARGET_EE_POS[0], TARGET_EE_POS[1], ee_pos[2].item() - 0.003],
                device=ee_pos.device,
                dtype=ee_pos.dtype,
            )

            # At pre_screw_ori (Rz(π)@Rx(π)): ee_x_dot_world_x = -1.
            # At intermediate    (Rz(π/2)@Rx(π)): ee_x_dot_world_x =  0.
            # At final target    (Rx(π)):          ee_x_dot_world_x = +1.
            if ee_x_dot_world_x < -0.25:
                # Phase 2a: first 90° CW (Rz: π → π/2).
                print("phase 2a")
                target_ori = C.rot_mat_tensor(np.pi, 0, np.pi / 2, device)[:3, :3]
            else:
                # Phase 2b: final 90° CW (Rz: π/2 → 0) to screw target.
                print("phase 2b")
                target_ori = C.rot_mat_tensor(np.pi, 0, 0 - 0.1, device)[:3, :3]

            # Intentially tilt a little bit into the corner of the obstacle to avoid the table_top sliding
            TILT_ANGLE = 3
            tilt = C.rot_mat_tensor(np.radians(TILT_ANGLE), np.radians(-TILT_ANGLE), 0, device)[:3, :3]
            target_ori = tilt @ target_ori

            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            target = self._add_noise_to_target(clean_target, pos_std=0.0, ori_std_deg=0.0)  # NO NOISE during screw
            result = self.satisfy(ee_pose, target, ori_error_threshold=0.3, max_len=75)
            if result == "TIMEOUT":
                timeout_failure = True

        skill_complete = self.state_transition_handler(self._last_state, state)

        return (
            target[:3, 3],
            C.mat2quat(target[:3, :3]),
            clean_target[:3, 3],
            C.mat2quat(clean_target[:3, :3]),
            torch.tensor([self.gripper_action], device=device),
            skill_complete,
            timeout_failure,
        )

    def current_state_no_noise(self):
        return self._last_state in self._NO_ACTION_NOISE_STATES

    def current_state_low_action_noise(self):
        return self._last_state in self._LOW_ACTION_NOISE_STATES

    def current_state_clean_action_noise(self):
        return self._last_state in self._CLEAN_ACTION_NOISE_STATES

    def _find_closest_y(self, pose):
        closest_y = pose.clone()
        for i in range(4):
            tmp_pose = pose @ torch.tensor(self.rel_pose_from_center[self.tag_ids[i]]).float().to(pose.device)
            if tmp_pose[1, 3] < closest_y[1, 3]:
                closest_y = tmp_pose
        return closest_y
