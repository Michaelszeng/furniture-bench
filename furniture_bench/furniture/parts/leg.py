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
        ]  # Specificy next state after skill is complete. Screw done is handle in `get_assembly_action`

        self.reset()

        self.part_attached_skill_idx = 4

    def reset(self):
        super().reset()  # resets prev_cnt=0, curr_cnt=0, pre_assemble_done, etc.
        self._last_state = "reach_leg_floor_xy"
        self.gripper_action = -1
        self.screw_mode = "standard"  # latent variable: "standard" or "alternate" (-90° Z offset)
        self.latent_offsets = {}  # per-state 3D position offsets, empty = no latent plan

    def apply_non_markovian_config(self):
        """Sample episode-level non-Markovian latent variables."""
        import random

        import numpy as np

        self.screw_mode = random.choice(["standard", "alternate"])

        # States that require positional precision get small offsets; all others get large offsets.
        PRECISION_STATES = {"reach_leg_floor_z", "reach_table_top_z", "screw_grasp", "screw"}
        HIGH_STD = 0.020  # m — persistent offset for coarse-motion states
        LOW_STD = 0.003  # m — persistent offset for precision states

        all_states = [
            "reach_leg_floor_xy",
            "reach_leg_ori",
            "reach_leg_floor_z",
            "pick_leg",
            "lift_up",
            "match_leg_ori",
            "reach_table_top_xy",
            "reach_table_top_z",
            "insert_release",
            "release",
            "pre_screw",
            "screw_grasp",
            "screw",
        ]
        self.latent_offsets = {
            state: np.random.normal(0, LOW_STD if state in PRECISION_STATES else HIGH_STD, size=(3,))
            for state in all_states
        }

    def _apply_latent_offset(self, state: str, clean_target):
        """
        Add the episode-level latent position offset to clean_target[:3, 3].
        If self.latent_offsets is not set, no-op.
        """
        if not self.latent_offsets:
            return clean_target
        offset = self.latent_offsets.get(state)
        if offset is None:
            return clean_target
        clean_target = clean_target.clone()
        clean_target[:3, 3] += torch.tensor(offset, device=clean_target.device, dtype=clean_target.dtype)
        return clean_target

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

        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))  # EE Pose in robot base frame
        table_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )  # TEMPORARY: Table pose in world frame
        leg_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )  # TEMPORARY: Leg pose in world frame
        table_pose = sim_to_april_mat @ table_pose  # Table pose in robot base frame
        leg_pose = sim_to_april_mat @ leg_pose  # Leg pose in robot base frame

        table_pose_robot = april_to_robot @ table_pose
        leg_pose_robot = april_to_robot @ leg_pose

        # Table hole pose in robot base frame
        table_hole_pose_robot = (
            april_to_robot
            @ table_pose
            @ torch.tensor(
                get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                device=device,
            ).float()
        )
        table_hole_pos_robot = table_hole_pose_robot[:3, 3]

        # Top-level phase discriminators
        leg_xy_near_hole = (leg_pose_robot[:2, 3] - table_hole_pos_robot[:2]).abs().sum() < 0.015
        # gripper_grasped: leg is physically between the fingers.
        # When the leg (diameter = 2*half_width) is held, physics prevents the gripper
        # from closing below 2*half_width, so the reading stays near that value.
        # Two-sided bound:
        #   upper: gripper hasn't opened past the leg (not released)
        #   lower: gripper isn't empty-closed (width ≈ 0 when nothing is held)
        # A 0.005 m upper margin and 0.010 m lower margin accommodate measurement noise.
        leg_diameter = 2 * self.half_width
        gripper_grasped = (
            gripper_width < leg_diameter + 0.005  # not open
            and gripper_width > leg_diameter - 0.010  # not empty-closed
        )

        # Precompute orientations
        margin = C.rot_mat_tensor(0, -np.pi / 5, 0, device)  # Ry(-36 deg)
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
        _grasp_base = (margin @ april_to_robot @ C.rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device))[:3, :3]
        _Rz = torch.tensor(
            [
                [np.cos(_theta_wrapped), -np.sin(_theta_wrapped), 0.0],
                [np.sin(_theta_wrapped), np.cos(_theta_wrapped), 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        grasp_ori = _Rz @ _grasp_base
        insert_ori = (margin @ C.rot_mat_tensor(np.pi, 0, 0, device))[
            :3, :3
        ]  # Ry(-36°) @ Rx(180°) — matches match_leg_ori target
        table_top_z = table_pose_robot[2, 3]
        insert_z = table_top_z + 0.056

        # Hard-coded INSERTED-phase orientations (mode-dependent).
        # Standard:  pre_screw = Rx(π)@Rz(π),   screw_done = Rx(π)           (−180° around Z)
        # Alternate: pre_screw = Rx(π)@Rz(π/2), screw_done = Rx(π)@Rz(−π/2) (same arc, −90° offset)
        if self.screw_mode == "alternate":
            pre_screw_ori = C.rot_mat_tensor(np.pi, 0, np.pi / 2, device)[:3, :3]
            screw_ori = C.rot_mat_tensor(np.pi, 0, -np.pi / 2, device)[:3, :3]
        else:
            pre_screw_ori = C.rot_mat_tensor(np.pi, 0, np.pi, device)[:3, :3]
            screw_ori = C.rot_mat_tensor(np.pi, 0, 0, device)[:3, :3]

        # Stable pre-screw position: table hole XY + insert_z height.
        # Using the stable table reference (not the bouncing leg) avoids the EE chasing
        # the leg downward immediately after insertion.
        pre_screw_pos = table_hole_pos_robot[:3].clone()
        pre_screw_pos[2] = insert_z
        at_pre_screw_pos = (ee_pos - pre_screw_pos).abs().sum() < self.pos_error_threshold * 3

        # ── Phase 1: GRASPED ─────────────────────────────────────────────────
        # Primary discriminator: leg is physically between the gripper fingers.
        if gripper_grasped:
            if leg_xy_near_hole:
                # ── Screw vs Insertion sub-phase ─────────────────────────────
                # Both "carrying the leg down for insertion" and "re-grasped for screwing"
                # share the same XY proximity to the hole.  The EE orientation is the discriminator:
                #   insert_ori (Ry(-36°) @ Rx(180°)) → still inserting
                #   anything else                     → screw sub-phase
                ee_at_insert_ori = (ee_pose[:3, :3] - insert_ori).abs().sum() < self.ori_error_threshold * 2
                if ee_at_insert_ori:
                    # ── Insertion sub-phase ──────────────────────────────────
                    # EE is at insert_ori, descending to insert the leg.
                    print(
                        f"leg_pose_robot[2, 3] - table_pose_robot[2, 3]: {leg_pose_robot[2, 3] - table_pose_robot[2, 3]}"
                    )
                    if leg_pose_robot[2, 3] - table_pose_robot[2, 3] < 0.056:  # fully inserted → release the leg
                        return "insert_release"
                    return "reach_table_top_z"  # descending to insert the leg

                else:
                    # ── Screw sub-phase ──────────────────────────────────────
                    # EE is rotating from pre_screw_ori toward screw_ori.
                    # screwing complete (EE has rotated to screw_ori) → release
                    if (ee_pose[:3, :3] - screw_ori).abs().sum() < self.ori_error_threshold * 1.5:
                        return "release"
                    return "screw"

            else:
                # ── Transport sub-phase ──────────────────────────────────────
                # Leg is grasped and the EE is not yet over the hole.
                # Navigate in order: lift → center → orient → fly to hole.

                # "reach_table_top_xy": EE orientation already matches insert_ori
                # (Ry(-36°) @ Rx(180°)) → fly EE XY to the hole position.
                if (ee_pose[:3, :3] - insert_ori).abs().sum() < self.ori_error_threshold * 3:
                    return "reach_table_top_xy"

                # "match_leg_ori": leg is lifted (Z > 0.05 m) → move to the
                # staging position [0.57, 0.1, 0.12] and rotate to insert_ori.
                if leg_pose_robot[2, 3] > 0.05:
                    return "match_leg_ori"

                # "lift_up": default — leg just grasped and still near the
                # floor (leg Z ≤ 0.05 m) → lift until leg clears the floor.
                return "lift_up"

        # ── Phase 2: INSERTED ────────────────────────────────────────────────
        # Primary discriminator: leg XY is near the hole AND gripper is open.
        # The leg was released into the hole.
        # Screw sub-sequence: pre_screw → screw_grasp → screw.
        elif leg_xy_near_hole:
            d_from_pre_screw = (ee_pose[:3, :3] - pre_screw_ori).abs().sum()

            # "screw_grasp": at pre_screw_ori AND at position → close gripper.
            if d_from_pre_screw < self.ori_error_threshold * 1.5 and at_pre_screw_pos:
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
            leg_xy = leg_pose_robot[:2, 3]
            leg_z = leg_pose_robot[2, 3]

            # Gate on XY proximity between EE and leg.
            if (ee_pos[:2] - leg_xy).abs().sum() < self.pos_error_threshold * 4:
                # Gate on orientation alignment between EE and leg.
                if (ee_pose[:3, :3] - grasp_ori).abs().sum() < self.ori_error_threshold * 1.5:
                    # Gate on Z proximity between EE and leg.
                    if abs(ee_pos[2] - leg_z) < self.pos_error_threshold * 2:
                        return "pick_leg"  # Ready to pick up the leg.

                    return "reach_leg_floor_z"  # EE is at leg XY + grasp_ori but above the leg → descend to leg Z.

                return "reach_leg_ori"  # EE is close to leg in XY but not yet aligned with grasp_ori → rotate in place.

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

        margin = C.rot_mat_tensor(0, -np.pi / 5, 0, ee_pose.device)
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
            target_pos[2] = ee_pos[2]
            # target_pos[1] += 0.01
            print(f"grasp_margin_x: {0.005}")
            target_pos[0] += 0.005
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            target = self._add_noise(clean_target, pos_std=0.005, ori_std_deg=5.0)
            # Large XY motion from neutral to leg (can be 10-15 cm); needs many steps.
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.02, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "reach_leg_ori":
            leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
            target_ori = floor_grasp_ori(leg_pose_down)
            leg_pos_robot = (april_to_robot @ leg_pose_down[:4, 3])[:3]
            target_pos = leg_pos_robot.clone()
            target_pos[2] = ee_pos[2]
            # target_pos[1] += 0.01
            target_pos[0] += 0.005
            target_pos[2] -= 0.005
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            target = self._add_noise(clean_target, pos_std=0.003, ori_std_deg=3.0)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.015, ori_error_threshold=0.1, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "reach_leg_floor_z":
            leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
            target_ori = floor_grasp_ori(leg_pose_down)
            leg_pos_robot = (april_to_robot @ leg_pose_down[:4, 3])[:3]
            target_pos = leg_pos_robot.clone()
            # target_pos[1] += 0.01
            target_pos[0] += 0.005
            target_pos[2] = (april_to_robot @ leg_pose)[2, 3]
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            target = self._add_noise(clean_target, pos_std=0.0015, ori_std_deg=1.5)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.015, ori_error_threshold=0.3, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "pick_leg":
            leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
            target_ori = floor_grasp_ori(leg_pose_down)
            leg_pos_robot = (april_to_robot @ leg_pose_down[:4, 3])[:3]
            target_pos = leg_pos_robot.clone()
            # target_pos[1] += 0.01
            target_pos[0] += 0.005
            target_pos[2] = (april_to_robot @ leg_pose)[2, 3]
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            target = self._add_noise(clean_target, pos_std=0.003, ori_std_deg=3.0)
            self.gripper_action = 1
            result = self.gripper_less(gripper_width, 2 * self.half_width + 0.001)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "lift_up":
            leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
            leg_pos_robot = (april_to_robot @ leg_pose_down[:4, 3])[:3]
            target_pos = leg_pos_robot.clone()
            target_pos[2] += 0.10
            target_ori = ee_pose[:3, :3]
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            target = self._add_noise(clean_target)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "match_leg_ori":
            target_ori = (margin @ C.rot_mat_tensor(np.pi, 0, 0, device))[:3, :3]
            target_pos = torch.tensor([0.57, 0.1, 0.12], device=device)
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            target = self._add_noise(clean_target)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "reach_table_top_xy":
            leg_pose_robot = april_to_robot @ leg_pose
            leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
            table_hole_pose_robot = (
                april_to_robot
                @ table_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_leg_pose_robot = torch.tensor(
                [  # 0.004 and 0.003 are empirical offsets to make the leg align perfectly with the table hole
                    [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3] + 0.004],
                    [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3] + 0.003],
                    [0.0, 1.0, 0.0, table_pose[2, 3] + 0.14],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            rel = rel_rot_mat(leg_pose_robot, target_leg_pose_robot)
            clean_target = rel @ ee_pose
            clean_target = self._apply_latent_offset(state, clean_target)
            target = self._add_noise(clean_target, pos_std=0.003, ori_std_deg=3.0)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.015, ori_error_threshold=0.3, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "reach_table_top_z":
            leg_pose_robot = april_to_robot @ leg_pose
            leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
            table_hole_pose_robot = (
                april_to_robot
                @ table_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_leg_pose_robot = torch.tensor(
                [  # 0.004 and 0.003 are empirical offsets to make the leg align perfectly with the table hole
                    [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3] + 0.004],
                    [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3] + 0.003],
                    [0.0, 1.0, 0.0, table_pose[2, 3] + 0.05],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            rel = rel_rot_mat(leg_pose_robot, target_leg_pose_robot)
            clean_target = rel @ ee_pose
            clean_target = self._apply_latent_offset(state, clean_target)
            target = self._add_noise(clean_target, pos_std=0.001, ori_std_deg=1.0)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.007, ori_error_threshold=0.15, max_len=75)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "insert_release":
            self.gripper_action = -1
            # Sub-phase gate: hold EE in place until the gripper has cleared the leg,
            # then move upward to the pre-screw position.
            gripper_cleared_leg = gripper_width >= 2 * self.half_width + 0.010
            if gripper_cleared_leg:
                leg_pose_robot = april_to_robot @ leg_pose
                leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
                table_hole_pose_robot = (
                    april_to_robot
                    @ table_pose
                    @ torch.tensor(
                        get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                        device=device,
                    )
                )
                target_leg_pose_robot = torch.tensor(
                    [  # 0.004 and 0.003 are empirical offsets to make the leg align perfectly with the table hole
                        [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3] + 0.004],
                        [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3] + 0.003],
                        [0.0, 1.0, 0.0, table_pose[2, 3] + 0.084],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    device=device,
                )
                rel = rel_rot_mat(leg_pose_robot, target_leg_pose_robot)
                clean_target = rel @ ee_pose
                clean_target = self._apply_latent_offset(state, clean_target)
                target = self._add_noise(clean_target, pos_std=0.001, ori_std_deg=1.0)
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
            target = self._add_noise(clean_target, pos_std=0.001, ori_std_deg=1.0)
            self.gripper_action = -1
            result = self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["square_table"] - 0.001,
            )
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "pre_screw":
            target_pos = (april_to_robot @ leg_pose)[:3, 3].clone()
            target_pos[2] = 0.055

            ee_z_dot_down = -ee_pose[2, 2]  # 1.0 = EE Z-axis straight down
            ee_x_dot_world_x = ee_pose[0, 0]  # X-component of EE local-X (world frame)

            # 2-stage sequence: first right the gripper (so it is vertical); then, rotate +180 deg about
            # world-Z to pre_screw_ori.
            if ee_z_dot_down < 0.99:
                # Phase 1: right the gripper (same for both modes).
                target_ori = C.rot_mat_tensor(np.pi, 0, 0, device)[:3, :3]
            elif self.screw_mode == "alternate":
                ee_x_dot_world_y = ee_pose[1, 0]
                if ee_x_dot_world_y < 0.2:
                    # Phase 2a: rotate to Rz(30°) first.
                    print("phase 2a")
                    target_ori = C.rot_mat_tensor(np.pi, 0, np.pi / 4, device)[:3, :3]
                else:
                    # Phase 2b: final 90° CCW to Rx(π)@Rz(π/2).
                    print("phase 2b")
                    target_ori = C.rot_mat_tensor(np.pi, 0, np.pi / 2, device)[:3, :3]
            else:  # standard
                # Standard mode: 180° rotation split into two 90° phases to avoid ambiguity.
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
            target = self._add_noise(clean_target, pos_std=0.0, ori_std_deg=0.0)
            result = self.satisfy(ee_pose, target, max_len=300)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "screw_grasp":
            # IMPORTANT: define target relative to leg pose so that we grab it centered
            target_pos = (april_to_robot @ leg_pose)[:3, 3].clone()
            # target_pos[2] = table_pose_robot[2, 3] + 0.065
            target_pos[2] = 0.055
            if self.screw_mode == "alternate":
                target_ori = C.rot_mat_tensor(np.pi, 0, np.pi / 2, device)[:3, :3]
            else:  # standard
                target_ori = C.rot_mat_tensor(np.pi, 0, np.pi, device)[:3, :3]
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            target = self._add_noise(clean_target, pos_std=0.0, ori_std_deg=0.0)
            self.gripper_action = 1
            result = self.gripper_less(gripper_width, 2 * self.half_width + 0.001)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "screw":
            # Rotate EE -180° about global Z from pre_screw back to Rx(π).
            # Split into two 90° steps (same technique as pre_screw) to avoid rotational ambiguity.
            ee_x_dot_world_x = ee_pose[0, 0]
            # # Anchor XY to the table hole (stable external reference) to provide a restoring
            # # force against any XY perturbations (e.g. table_top sliding during screwing).
            # table_hole_pose_robot = (
            #     april_to_robot
            #     @ table_pose
            #     @ torch.tensor(
            #         get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
            #         device=device,
            #     ).float()
            # )
            # target_pos = table_hole_pose_robot[:3, 3].clone()
            # target_pos[2] = ee_pos[2] - 0.005  # keep downward pressure while screwing

            # Simply hard-code target position during screwing
            # Note: intentionally make the target height during screwing 1mm above the height during screw_grasp
            # This way, the gripper doesn't apply too much force into the table_top
            # ACTUALLY: the downward pressure works well too, but it makes the dynamics a little more predictable.
            TARGET_EE_POS = [0.6453, 0.1375]
            target_pos = torch.tensor(
                [TARGET_EE_POS[0], TARGET_EE_POS[1], 0.056],
                # [TARGET_EE_POS[0], TARGET_EE_POS[1], ee_pos[2].item() - 0.005],
                device=ee_pos.device,
                dtype=ee_pos.dtype,
            )
            # target_pos = ee_pos[:3].clone()
            # target_pos[2] = 0.055
            # target_pos[2] += 0.001
            # target_pos[2] = ee_pos[2] - 0.004  # keep downward pressure while screwing
            print(f"target_pos: {target_pos}")

            # print("abs(ee_pos[0] - TARGET_EE_POS[0]):", abs(ee_pos[0] - TARGET_EE_POS[0]))
            # print("abs(ee_pos[1] - TARGET_EE_POS[1]):", abs(ee_pos[1] - TARGET_EE_POS[1]))

            # if abs(ee_pos[0] - TARGET_EE_POS[0]) > 0.0025 or abs(ee_pos[1] - TARGET_EE_POS[1]) > 0.0025:
            #     current_yaw = torch.atan2(ee_pose[1, 0], ee_pose[0, 0])
            #     target_ori = C.rot_mat_tensor(np.pi, 0, current_yaw.item(), device)[:3, :3]
            # else:
            if self.screw_mode == "alternate":
                ee_x_dot_world_y = ee_pose[1, 0]
                if ee_x_dot_world_y > 0.75:
                    # Phase a: first 90° CW (Rz: π/2 → 0).
                    print("phase 2a")
                    target_ori = C.rot_mat_tensor(np.pi, 0, 0, device)[:3, :3]
                else:
                    # Phase b: second 90° CW (Rz: 0 → −π/2).
                    print("phase 2b")
                    target_ori = C.rot_mat_tensor(np.pi, 0, -np.pi / 2 - 0.1, device)[:3, :3]
            else:  # standard
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
            target = self._add_noise(clean_target, pos_std=0.0, ori_std_deg=0.0)
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

    def state_no_noise(self):
        return self._last_state in [
            "insert_release",
        ]

    def _find_closest_y(self, pose):
        closest_y = pose.clone()
        for i in range(4):
            tmp_pose = pose @ torch.tensor(self.rel_pose_from_center[self.tag_ids[i]]).float().to(pose.device)
            if tmp_pose[1, 3] < closest_y[1, 3]:
                closest_y = tmp_pose
        return closest_y
