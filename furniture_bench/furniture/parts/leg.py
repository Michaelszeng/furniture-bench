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
    _NM_PAUSE_EXCLUDED_STATES: frozenset = frozenset(
        {"release", "lift_up", "reach_leg_floor_xy", "reach_leg_floor_z", "insert"}
    )

    # ── Per-state noise tiers ──────────────────────────────────────────────────
    # Target-noise tiers (used by apply_non_markovian_config for latent offsets)
    _LOW_LATENT_TARGET_STD_STATES: frozenset = frozenset(
        {"reach_leg_floor_z", "pick_leg", "reach_table_top_z", "release"}
    )
    _LOW_LATENT_TARGET_Z_STD_STATES: frozenset = frozenset(
        {
            "pick_leg",
            "lift_up",
            "reach_table_top_xy",
            "reach_table_top_z",
            "pre_screw",
            "release",
        }
    )
    _ZERO_LATENT_TARGET_POS_STD_STATES: frozenset = frozenset(
        {"reach_leg_ori", "reach_leg_floor_z", "pre_screw", "screw_grasp", "screw", "insert_release", "insert"}
    )
    _ZERO_LATENT_TARGET_ORI_STD_STATES: frozenset = frozenset(
        {"reach_leg_ori", "pre_screw", "screw_grasp", "screw", "insert_release", "insert"}
    )

    # Action-noise tiers (used by furniture_sim_env when computing the executed action).
    _LOW_ACTION_NOISE_STATES: frozenset = frozenset({"reach_table_top_z", "pre_screw", "screw_grasp", "screw"})
    _NO_ACTION_NOISE_STATES: frozenset = frozenset({"insert_release", "insert"})

    # Virtual-target walk noise tiers (used by furniture_sim_env).
    _LOW_RANDOM_WALK_NOISE_STD_STATES: frozenset = frozenset({"reach_table_top_z"})
    _ZERO_RANDOM_WALK_NOISE_Z_STD_STATES: frozenset = frozenset({"reach_table_top_z"})
    _ZERO_RANDOM_WALK_NOISE_STD_STATES: frozenset = frozenset(
        {
            "reach_leg_floor_z",
            "pick_leg",
            "pre_screw",
            "screw_grasp",
            "screw",
            "insert_release",
            "insert",
        }
    )

    # States where the noisy (executed) action is also recorded as the clean action.
    _CLEAN_ACTION_NOISE_STATES: frozenset = frozenset({"reach_table_top_z"})

    # ── NM screw-grasp alignment cycles ──────────────────────────────────────
    # Before closing the gripper, the EE makes 1..MAX_CYCLES random target shifts
    # (mimicking a human repositioning to align) followed by one final clean cycle.
    _NM_SCREW_GRASP_MIN_ALIGN_CYCLES: int = 1
    _NM_SCREW_GRASP_MAX_ALIGN_CYCLES: int = 3  # max random cycles (final clean cycle is always added)
    _NM_SCREW_GRASP_ALIGN_STEPS_MIN: int = 3  # min steps per cycle
    _NM_SCREW_GRASP_ALIGN_STEPS_MAX: int = 9  # max steps per cycle
    _NM_SCREW_GRASP_ALIGN_CLEAN_CYCLE_EXTRA_STEPS: int = (
        12  # extra steps added to the final clean cycle for VT convergence
    )
    _NM_SCREW_GRASP_ALIGN_POS_STD: float = 0.008  # std of random XY offset (m)

    # ── NM reach_leg_floor_z staged alignment ────────────────────────────────
    # Before reaching the final pick pose, the EE makes 1..MAX_CYCLES approaches
    # from progressively smaller positive-x / positive-z offsets, with a random y
    # jitter each cycle.  The final cycle uses the clean target with no offset.
    _NM_REACH_LEG_FLOOR_Z_MIN_ALIGN_CYCLES: int = 2
    _NM_REACH_LEG_FLOOR_Z_MAX_ALIGN_CYCLES: int = 4  # max staged cycles (final clean cycle always added)
    _NM_REACH_LEG_FLOOR_Z_ALIGN_STEPS_MIN: int = 8  # min steps per cycle
    _NM_REACH_LEG_FLOOR_Z_ALIGN_STEPS_MAX: int = 12  # max steps per cycle
    _NM_REACH_LEG_FLOOR_Z_ALIGN_X_OFFSET_MAX: float = 0.03  # x offset (m) at cycle 0, ramps to 0
    _NM_REACH_LEG_FLOOR_Z_ALIGN_Z_OFFSET_MAX: float = 0.03  # z offset (m) at cycle 0, ramps to 0
    _NM_REACH_LEG_FLOOR_Z_ALIGN_X_STD: float = 0.02  # std of random x jitter per cycle (m)
    _NM_REACH_LEG_FLOOR_Z_ALIGN_Y_STD: float = 0.014  # std of random y jitter per cycle (m)
    _NM_REACH_LEG_FLOOR_Z_ALIGN_Z_STD: float = 0.01  # std of random z jitter per cycle (m)

    # ── NM insertion descent pauses ───────────────────────────────────────────
    # During reach_table_top_z, the robot may randomly pause its descent to mimic
    # a human double-checking alignment before committing to insertion.
    _NM_INSERTION_PAUSE_PROB: float = 0.2  # per-step probability of starting a pause
    _NM_INSERTION_PAUSE_STEPS_MIN: int = 3
    _NM_INSERTION_PAUSE_STEPS_MAX: int = 8

    # ── Sticky transition delays ───────────────────────────────────────────
    _NM_STICKY_REACH_LEG_FLOOR_Z_PICK_LEG_MIN_DELAY: int = 10
    _NM_STICKY_REACH_LEG_FLOOR_Z_PICK_LEG_MAX_DELAY: int = 20

    _NM_STICKY_PICK_LEG_LIFT_UP_MIN_DELAY: int = 5
    _NM_STICKY_PICK_LEG_LIFT_UP_MAX_DELAY: int = 10

    _NM_STICKY_PRE_SCREW_SCREW_GRASP_MIN_DELAY: int = 5
    _NM_STICKY_PRE_SCREW_SCREW_GRASP_MAX_DELAY: int = 10

    _NM_STICKY_REACH_TABLE_TOP_Z_INSERT_MIN_DELAY: int = 5  # linger at alignment before committing to insert
    _NM_STICKY_REACH_TABLE_TOP_Z_INSERT_MAX_DELAY: int = 12

    _NM_STICKY_SCREW_RELEASE_MIN_DELAY: int = 4
    _NM_STICKY_SCREW_RELEASE_MAX_DELAY: int = 9

    # Ry angle (radians) that pitches the EE toward the floor during the floor pick-up.
    # Adjust here to change the grasp tilt; used identically in compute_state and fsm_step.
    _GRASP_MARGIN_ANGLE: float = -np.pi / 7
    _INSERT_TIP_RY_ANGLE: float = np.radians(4.5)  # slight y-axis tilt during table-top approach and insertion
    _LEG_TIP_OFFSET: float = 0.05625  # distance from leg mesh origin to screw tip (m)
    _LEG_HOLE_OFFSET_X: float = 0.0025  # fine-alignment offset of tip to table hole, X (m)
    _LEG_HOLE_OFFSET_Y: float = 0.001  # fine-alignment offset of tip to table hole, Y (m)
    _PICK_Z_OFFSET: float = 0.012  # EE hovers 1.2 cm above the leg COM during floor pick
    _INSERT_HANDOFF_Z: float = 0.025  # z distance from the table to the leg tip when ready to insert

    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)

        self._PICK_X_OFFSET: float = 0.002 if self.non_markovian else 0.005  # grab 0.5 cm toward "top" of leg along X
        self._ORI_Z_CLEARANCE: float = 0.13 if self.non_markovian else 0.05  # Z clearance during orientation alignment

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
        # Shared floor-pick cache: leg_pose_down in the april frame, captured on first entry to
        # reach_leg_floor_xy and reused through reach_leg_ori, reach_leg_floor_z, and pick_leg.
        # Prevents the target from drifting if EE contact nudges the leg during approach.
        self.nm_floor_pick_cached_leg_pose_down = None
        # Cached EE z at the start of reach_leg_floor_xy. Used as the fixed base for the z-target
        # so the latent z offset doesn't compound each step (EE chasing its own drifting position).
        self.nm_reach_leg_floor_xy_cached_ee_z: float | None = None
        # Screw-grasp alignment state (NM only).
        self.nm_screw_grasp_align_cycles_remaining = None  # None = uninitialised
        self.nm_screw_grasp_align_done = False
        self.nm_screw_grasp_align_step_end = 0
        self.nm_screw_grasp_align_offset = None
        # reach_leg_floor_z staged alignment state (NM only).
        self.nm_reach_leg_floor_z_total_cycles: int = 0  # sampled once on state entry
        self.nm_reach_leg_floor_z_align_cycles_remaining = None
        self.nm_reach_leg_floor_z_align_done = False
        self.nm_reach_leg_floor_z_align_step_end = 0
        self.nm_reach_leg_floor_z_align_offset = None
        # Insertion descent pause state (NM only).
        self.nm_insertion_pause_step_end: int = 0
        self.nm_insertion_pause_cached_z: float = 0.0
        self.leg_tip_z_rel: float = float("inf")  # set each step by compute_state
        self.gripper_action = -1

    def apply_non_markovian_config(self):
        """Sample episode-level non-Markovian latent variables."""
        if not self._NM_LATENT_PLAN:
            return

        HIGH_STD = 0.007  # m — persistent position offset for coarse-motion states
        LOW_STD = 0.005  # m — persistent position offset for precision states
        HIGH_ORI_STD = np.radians(3.0)  # rad — persistent orientation offset for coarse-motion states
        LOW_ORI_STD = np.radians(2.0)  # rad — persistent orientation offset for precision states

        all_states = [
            "reach_leg_floor_xy",
            "reach_leg_ori",
            "reach_leg_floor_z",
            "pick_leg",
            "lift_up",
            "match_leg_ori",
            "reach_table_top_xy",
            "reach_table_top_z",
            "insert",
            "insert_release",
            "release",
            "pre_screw",
            "screw_grasp",
            "screw",
        ]

        def pos_std_for(state):
            """Return a (3,) array of per-axis position stds [x, y, z].

            XY and Z axes are controlled independently:
              XY: ZERO_POS → 0, LOW → LOW_STD, else HIGH_STD
              Z:  ZERO_POS → 0, LOW_Z → LOW_STD, else HIGH_STD
            A state can appear in both LOW and LOW_Z to get LOW_STD on all axes.
            """
            if state in self._ZERO_LATENT_TARGET_POS_STD_STATES:
                return np.zeros(3)
            xy_std = LOW_STD if state in self._LOW_LATENT_TARGET_STD_STATES else HIGH_STD
            z_std = LOW_STD if state in self._LOW_LATENT_TARGET_Z_STD_STATES else HIGH_STD
            return np.array([xy_std, xy_std, z_std])

        def ori_std_for(state):
            if state in self._ZERO_LATENT_TARGET_ORI_STD_STATES:
                return 0.0
            if state in self._LOW_LATENT_TARGET_STD_STATES:
                return LOW_ORI_STD
            return HIGH_ORI_STD

        self.latent_offsets = {
            state: {
                "pos": np.random.normal(0, pos_std_for(state)),  # per-axis std via broadcast
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

        _leg_tip_xy = leg_pose_robot[:2, 3] - leg_pose_robot[:2, 1] * self._LEG_TIP_OFFSET

        _leg_diameter = 2 * self.half_width

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

        # Staging position during "match_leg_ori" and before "reach_table_top_xy"
        self.staging_pos = torch.tensor([0.45, 0.15, 0.14], device=device)

        ################################################################################################################
        # SHARED CONDITIONS
        # lo_t uses _last_state's latent offset; _lo() returns zeros when non_markovian=False
        # (latent_offsets is never populated), so all booleans below are correct for both paths.
        ################################################################################################################
        lo_t = torch.tensor(self._lo(), dtype=ee_pos.dtype, device=device)

        # ON_FLOOR floor-pick approach conditions (lo_t shifts targets to match the latent-offset target tracked in fsm_step)
        at_pick_xy = (ee_pos[:2] - (pick_target_xy + lo_t[:2])).abs().sum() < self.pos_error_threshold * 3
        # Gate on orientation alignment between EE and leg.
        # Randomized gate to get some grasp angle variation.
        at_grasp_ori_floor = (ee_pose[:3, :3] - grasp_target_ori).abs().sum() < self.ori_error_threshold * 3
        at_pick_z = abs(ee_pos[2] - (pick_target_z + lo_t[2])) < self.pos_error_threshold

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

        # Transport sub-phase: leg is clear of the floor after lift_up.
        leg_lifted = leg_pose_robot[2, 3] > 0.05
        # For safety, "match_leg_ori" is only returned when EE is still near the staging position;
        # once the EE has flown far toward the hole, we should not come back to match_leg_ori.
        near_staging = (ee_pos[:2] - self.staging_pos[:2]).abs().sum() < 0.2

        # Top-level phase discriminators
        # Use leg tip (screw threads) rather than COM for the XY proximity check.
        # Tip is LEG_TIP_OFFSET m in the -local-Y direction from the mesh origin.
        if self.non_markovian:
            leg_xy_near_hole = torch.norm(_leg_tip_xy - table_hole_pos_robot[:2]) < 0.015  # looser for NM
            leg_xy_near_hole_loose = torch.norm(_leg_tip_xy - table_hole_pos_robot[:2]) < 0.020
        else:
            leg_xy_near_hole = torch.norm(_leg_tip_xy - table_hole_pos_robot[:2]) < 0.009
            leg_xy_near_hole_loose = torch.norm(_leg_tip_xy - table_hole_pos_robot[:2]) < 0.015
        # Strict XY alignment required to hand off from reach_table_top_z to insert.
        leg_xy_aligned_strict = torch.norm(_leg_tip_xy - table_hole_pos_robot[:2]) < 0.003
        # Transition to insert when the leg tip is within 4 cm of the table AND XY is tightly aligned.
        ready_for_insert = leg_tip_z_rel < self._INSERT_HANDOFF_Z and leg_xy_aligned_strict

        # EE orientation checks (used across multiple GRASPED sub-phases)
        ee_at_insert_ori = (ee_pose[:3, :3] - insert_target_ori).abs().sum() < self.ori_error_threshold * 3
        # Loose bound: used to decide when the EE has sufficiently rotated to proceed toward the hole.
        ee_at_insert_ori_loose = (ee_pose[:3, :3] - insert_target_ori).abs().sum() <= self.ori_error_threshold * (
            5 if self.non_markovian else 4
        )

        # Sometimes, due to action noise, leg_xy_near_hole may become untrue momentarily.
        # If the leg is still close to the hole and the robot is moving downward, we know we're probably still in
        # "reach_table_top_z".
        barely_missed_hole = leg_xy_near_hole_loose and leg_z_vel_robot < -0.06 and ee_at_insert_ori

        # Insertion depth: leg tip Z relative to table surface (in robot frame).
        leg_z_rel = leg_pose_robot[2, 3] - table_pose_robot[2, 3]
        leg_tip_z_rel = leg_z_rel - leg_pose_robot[2, 1] * self._LEG_TIP_OFFSET
        self.leg_tip_z_rel = leg_tip_z_rel
        leg_fully_inserted_z = leg_tip_z_rel < 0.05725 - self._LEG_TIP_OFFSET  # tip has cleared the hole threshold

        # INSERTED phase: EE at pre-screw position and orientation
        at_pre_screw_target_pos = (ee_pos - (pre_screw_target_pos + lo_t)).abs().sum() < self.pos_error_threshold * 3
        at_pre_screw_target_ori = (ee_pose[:3, :3] - pre_screw_target_ori).abs().sum() < self.ori_error_threshold * 1.5

        screw_done = (ee_pose[:3, :3] - screw_target_ori).abs().sum() < self.ori_error_threshold * 1.5

        # Insertion stuck detection — runs whenever the EE is in the insertion sub-phase (grasped, near hole,
        # at insert_ori).  Updates prev_ state variables and sets self.GOT_STUCK; does not return.
        # `stuck_this_step` is a LOCAL flag: True only when stuck fires on THIS step.
        # Both the NM and Markovian blocks use stuck_this_step (not self.GOT_STUCK) for the retry transition,
        # so the robot can re-enter reach_table_top_z on the very next step after being redirected.
        # self.GOT_STUCK remains True persistently for fsm_step noise handling.
        STUCK_Z_LOW = 0.01475  # Below this → leg is inserted; don't trigger stuck
        STUCK_Z_HIGH = 0.01675  # Above this → leg hasn't made progress into the hole
        stuck_this_step = False
        if not self.non_markovian and gripper_grasped and leg_xy_near_hole and ee_at_insert_ori:
            print(f"leg_tip_z_rel: {leg_tip_z_rel}, leg_z_vel_robot: {leg_z_vel_robot}")
            STUCK_Z_VEL_THRESHOLD = 0.025  # m/s — leg moving upward indicates it is trying the retry behavior
            # 2-step check: require that both the current and previous step are in the stuck zone
            z_range_stuck = leg_tip_z_rel < STUCK_Z_HIGH and self.prev_leg_tip_z_rel < STUCK_Z_HIGH
            self.prev_leg_tip_z_rel = leg_tip_z_rel
            z_vel_stuck = (
                leg_z_vel_robot > STUCK_Z_VEL_THRESHOLD
            )  # If robot is already moving upward, i.e. already backing out for retry, then continue doing so
            if (
                leg_tip_z_rel > STUCK_Z_LOW  # Don't trigger stuck if leg is inserted
                and leg_z_vel_robot > -0.05  # Leg is not moving downward
                and self.prev_leg_z_vel_robot > -0.05  # Leg was not moving downward in the previous step
                and (z_range_stuck or z_vel_stuck)
            ):
                self.GOT_STUCK = True
                stuck_this_step = True
                print("GOT_STUCK -- returning to reach_table_top_xy")
            self.prev_leg_z_vel_robot = leg_z_vel_robot
        else:
            # Not in insertion sub-phase — reset Markovian 2-step history so stale values don't persist.
            self.prev_leg_tip_z_rel = float("inf")
            self.prev_leg_z_vel_robot = float("inf")

        ################################################################################################################
        # NON-MARKOVIAN: sequential state machine — only check whether _last_state has been completed
        ################################################################################################################
        if self.non_markovian:
            current = self._last_state
            # Gripper fully open threshold (matches insert_release / release gripper_greater() target)
            gripper_open = gripper_width >= config["robot"]["max_gripper_width"]["square_table"] - 0.001

            # Deferred transition: the pause just completed — fire the pending state change now.
            pending = self._nm_pop_pending_transition()
            if pending is not None:
                return pending

            if current == "reach_leg_floor_xy" and at_pick_xy:
                # return self._nm_defer_transition_with_pause("reach_leg_ori")
                return "reach_leg_ori"
            elif current == "reach_leg_ori" and at_grasp_ori_floor:
                # return self._nm_defer_transition_with_pause("reach_leg_floor_z")
                return "reach_leg_floor_z"  # no pause on entry to reach_leg_floor_z
            elif current == "reach_leg_floor_z" and at_pick_z:
                if self._nm_sticky_delay(
                    "reach_leg_floor_z",
                    self._NM_STICKY_REACH_LEG_FLOOR_Z_PICK_LEG_MIN_DELAY,
                    self._NM_STICKY_REACH_LEG_FLOOR_Z_PICK_LEG_MAX_DELAY,
                ):
                    return current  # linger in "reach_leg_floor_z" before transitioning to "pick_leg"
                # return self._nm_defer_transition_with_pause("pick_leg")
                return "pick_leg"
            elif current == "pick_leg" and gripper_grasped:
                if self._nm_sticky_delay(
                    "pick_leg",
                    self._NM_STICKY_PICK_LEG_LIFT_UP_MIN_DELAY,
                    self._NM_STICKY_PICK_LEG_LIFT_UP_MAX_DELAY,
                ):
                    return current  # linger in "pick_leg" before transitioning to "lift_up"
                # return self._nm_defer_transition_with_pause("lift_up")
                return "lift_up"
            elif current == "lift_up" and leg_lifted:
                # return self._nm_defer_transition_with_pause("match_leg_ori")
                return "match_leg_ori"
            elif current == "match_leg_ori" and ee_at_insert_ori_loose:
                # return self._nm_defer_transition_with_pause("reach_table_top_xy")
                return "reach_table_top_xy"
            elif current == "reach_table_top_xy" and leg_xy_near_hole:
                return "reach_table_top_z"
            elif current == "reach_table_top_z":
                if ready_for_insert:
                    if self._nm_sticky_delay(
                        "reach_table_top_z",
                        self._NM_STICKY_REACH_TABLE_TOP_Z_INSERT_MIN_DELAY,
                        self._NM_STICKY_REACH_TABLE_TOP_Z_INSERT_MAX_DELAY,
                    ):
                        return current  # linger at alignment before committing to insert
                    return "insert"
            elif current == "insert":
                if leg_fully_inserted_z:
                    return "insert_release"
            elif current == "insert_release" and gripper_open:
                # return self._nm_defer_transition_with_pause("pre_screw")
                return "pre_screw"
            elif current == "pre_screw" and at_pre_screw_target_ori and at_pre_screw_target_pos:
                if self._nm_sticky_delay(
                    "pre_screw",
                    self._NM_STICKY_PRE_SCREW_SCREW_GRASP_MIN_DELAY,
                    self._NM_STICKY_PRE_SCREW_SCREW_GRASP_MAX_DELAY,
                ):
                    return current  # linger in "pre_screw" before transitioning to "screw_grasp"
                # return self._nm_defer_transition_with_pause("screw_grasp")
                return "screw_grasp"
            elif current == "screw_grasp" and gripper_grasped:
                # return self._nm_defer_transition_with_pause("screw")
                return "screw"
            elif current == "screw" and screw_done:
                if self._nm_sticky_delay(
                    "screw",
                    self._NM_STICKY_SCREW_RELEASE_MIN_DELAY,
                    self._NM_STICKY_SCREW_RELEASE_MAX_DELAY,
                ):
                    return current  # linger in "screw" before transitioning to "release"
                # return self._nm_defer_transition_with_pause("release")
                return "release"  # no pause on entry to release
            elif current == "release" and gripper_open:
                # return self._nm_defer_transition_with_pause("pre_screw")  # loops for each screw cycle
                return "pre_screw"  # loops for each screw cycle
            return current

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
                    # Stuck detection and prev_* updates already ran in the SHARED CONDITIONS block above.
                    if leg_fully_inserted_z:  # tip has cleared the hole threshold → release the leg
                        return "insert_release"
                    if stuck_this_step:
                        return "reach_table_top_xy"  # missed hole; retry approach
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
            # Re-entering reach_leg_floor_xy means the leg was dropped and may have moved;
            # clear the cache so the new leg position is captured on the first step.
            # Also clear when leaving pick_leg (sequence complete).
            if state == "reach_leg_floor_xy" or self._last_state == "pick_leg":
                self.nm_floor_pick_cached_leg_pose_down = None
                self.nm_reach_leg_floor_xy_cached_ee_z = None
            # Reset alignment state whenever we (re-)enter the relevant states.
            if state == "screw_grasp":
                self.nm_screw_grasp_align_cycles_remaining = None
                self.nm_screw_grasp_align_done = False
            if state == "reach_leg_floor_z":
                self.nm_reach_leg_floor_z_align_cycles_remaining = None
                self.nm_reach_leg_floor_z_align_done = False
            if state in ("reach_table_top_z", "insert"):
                self.nm_insertion_pause_step_end = 0

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
            if self.non_markovian:
                if self.nm_floor_pick_cached_leg_pose_down is None:
                    self.nm_floor_pick_cached_leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
                leg_pose_down = self.nm_floor_pick_cached_leg_pose_down
            else:
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
            if self.nm_reach_leg_floor_xy_cached_ee_z is None:
                self.nm_reach_leg_floor_xy_cached_ee_z = ee_pos[2].item()
            target_pos[2] = self.nm_reach_leg_floor_xy_cached_ee_z
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
            if self.non_markovian:
                if self.nm_floor_pick_cached_leg_pose_down is None:
                    self.nm_floor_pick_cached_leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
                leg_pose_down = self.nm_floor_pick_cached_leg_pose_down
            else:
                leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
            target_ori = floor_grasp_ori(leg_pose_down)
            leg_pos_robot = (april_to_robot @ leg_pose_down[:4, 3])[:3]
            target_pos = leg_pos_robot.clone()
            target_pos[0] = (
                leg_pos_robot[0]
                + self._PICK_X_OFFSET
                + (self._NM_REACH_LEG_FLOOR_Z_ALIGN_X_OFFSET_MAX if self.non_markovian else 0.0)
            )
            target_pos[2] = leg_pos_robot[2] + self._ORI_Z_CLEARANCE  # keep clearance above leg during EE rotation
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target,
                    pos_std=0.01,
                    ori_std_deg=2.0,
                    step_noise_pos_std=0.01,
                    step_noise_pos_std_z=0.004,
                    step_noise_ori_std_deg=0.5,
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.01, ori_std_deg=5)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.015, ori_error_threshold=0.1, max_len=60)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "reach_leg_floor_z":
            if self.non_markovian:
                self.set_speed(delta_pos_gain=1.2, max_delta_xy=0.01)

            if self.non_markovian:
                if self.nm_floor_pick_cached_leg_pose_down is None:
                    self.nm_floor_pick_cached_leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
                leg_pose_down = self.nm_floor_pick_cached_leg_pose_down
            else:
                leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
            target_ori = floor_grasp_ori(leg_pose_down)
            leg_pos_robot = (april_to_robot @ leg_pose_down[:4, 3])[:3]  # leg COM in the robot frame
            target_pos = leg_pos_robot.clone()
            target_pos[0] = leg_pos_robot[0] + self._PICK_X_OFFSET
            target_pos[2] = leg_pos_robot[2] + self._PICK_Z_OFFSET
            if self.non_markovian:
                # Staged alignment: EE approaches from a positive-x / positive-z offset that
                # decreases linearly each cycle.  A random y jitter is added per cycle.
                # The final clean cycle has no offset.  Mimics a human aiming in stages.
                def new_floor_z_cycle(cycle_idx, total):
                    frac = cycle_idx / total if total > 0 else 0.0
                    x_off = self._NM_REACH_LEG_FLOOR_Z_ALIGN_X_OFFSET_MAX * frac + float(
                        np.clip(
                            np.random.normal(0, self._NM_REACH_LEG_FLOOR_Z_ALIGN_X_STD),
                            -self._NM_REACH_LEG_FLOOR_Z_ALIGN_X_STD,
                            self._NM_REACH_LEG_FLOOR_Z_ALIGN_X_STD,
                        )
                    )
                    z_off = self._NM_REACH_LEG_FLOOR_Z_ALIGN_Z_OFFSET_MAX * frac + float(
                        np.clip(
                            np.random.normal(0, self._NM_REACH_LEG_FLOOR_Z_ALIGN_Z_STD),
                            -self._NM_REACH_LEG_FLOOR_Z_ALIGN_Z_STD,
                            self._NM_REACH_LEG_FLOOR_Z_ALIGN_Z_STD,
                        )
                    )
                    y_off = float(
                        np.clip(
                            np.random.normal(0, self._NM_REACH_LEG_FLOOR_Z_ALIGN_Y_STD),
                            -self._NM_REACH_LEG_FLOOR_Z_ALIGN_Y_STD * 0.9,  # Clip to -0.9, 0.9 std
                            self._NM_REACH_LEG_FLOOR_Z_ALIGN_Y_STD * 0.9,
                        )
                    )
                    self.nm_reach_leg_floor_z_align_offset = torch.tensor(
                        [x_off, y_off, z_off], dtype=target_pos.dtype, device=device
                    )
                    self.nm_reach_leg_floor_z_align_step_end = self.curr_cnt + np.random.randint(
                        self._NM_REACH_LEG_FLOOR_Z_ALIGN_STEPS_MIN, self._NM_REACH_LEG_FLOOR_Z_ALIGN_STEPS_MAX + 1
                    )

                if self.nm_reach_leg_floor_z_align_cycles_remaining is None:
                    self.nm_reach_leg_floor_z_total_cycles = np.random.randint(
                        self._NM_REACH_LEG_FLOOR_Z_MIN_ALIGN_CYCLES, self._NM_REACH_LEG_FLOOR_Z_MAX_ALIGN_CYCLES + 1
                    )
                    self.nm_reach_leg_floor_z_align_cycles_remaining = self.nm_reach_leg_floor_z_total_cycles
                    new_floor_z_cycle(
                        self.nm_reach_leg_floor_z_align_cycles_remaining, self.nm_reach_leg_floor_z_total_cycles
                    )
                    self.nm_reach_leg_floor_z_align_step_end += 1  # same +1 correction as pick_leg / screw_grasp

                if (
                    not self.nm_reach_leg_floor_z_align_done
                    and self.curr_cnt >= self.nm_reach_leg_floor_z_align_step_end
                ):
                    self.nm_reach_leg_floor_z_align_cycles_remaining -= 1
                    if self.nm_reach_leg_floor_z_align_cycles_remaining > 0:
                        new_floor_z_cycle(
                            self.nm_reach_leg_floor_z_align_cycles_remaining, self.nm_reach_leg_floor_z_total_cycles
                        )
                    elif self.nm_reach_leg_floor_z_align_cycles_remaining == 0:
                        new_floor_z_cycle(
                            0, self.nm_reach_leg_floor_z_total_cycles
                        )  # final clean cycle: frac=0 → no offset
                    else:
                        self.nm_reach_leg_floor_z_align_done = True

                pos = (
                    target_pos
                    if self.nm_reach_leg_floor_z_align_done
                    else target_pos + self.nm_reach_leg_floor_z_align_offset
                )
                clean_target = C.to_homogeneous(pos, target_ori)
                clean_target = self._apply_latent_offset(state, clean_target)
                target = self._add_noise_to_target(
                    clean_target,
                    pos_std=0.005,
                    ori_std_deg=2.5,
                    step_noise_pos_std=0.01,
                    step_noise_pos_std_z=0.004,
                    step_noise_ori_std_deg=2.5,
                )
                if not self.nm_reach_leg_floor_z_align_done:
                    print(
                        f"[reach_leg_floor_z NM align] cycles_left={self.nm_reach_leg_floor_z_align_cycles_remaining}"
                        f"  offset=({self.nm_reach_leg_floor_z_align_offset[0]:.3f},"
                        f" {self.nm_reach_leg_floor_z_align_offset[1]:.3f},"
                        f" {self.nm_reach_leg_floor_z_align_offset[2]:.3f})"
                    )
            else:
                clean_target = C.to_homogeneous(target_pos, target_ori)
                clean_target = self._apply_latent_offset(state, clean_target)
                target = self._add_noise_to_target(clean_target, pos_std=0.01, ori_std_deg=12.0)
            result = self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.015,
                ori_error_threshold=0.3,
                max_len=30 if not self.non_markovian else 120,
            )
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "pick_leg":
            if self.non_markovian:
                if self.nm_floor_pick_cached_leg_pose_down is None:
                    self.nm_floor_pick_cached_leg_pose_down = self._find_down_z(leg_pose).clone().to(device)
                leg_pose_down = self.nm_floor_pick_cached_leg_pose_down
            else:
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
                    clean_target, pos_std=0.001, ori_std_deg=1.0, step_noise_pos_std=0.0, step_noise_ori_std_deg=1.0
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
            if self.non_markovian:
                target_pos = self.staging_pos
            else:
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
        elif state == "reach_table_top_xy":
            target_z = 0.11 if self.non_markovian else 0.14
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
            # Skip latent offset on retry: the same episode-level XY bias that caused the miss
            # would otherwise keep the approach in the same wrong position on every retry.
            if not self.GOT_STUCK:
                clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target,
                    pos_std=0.002,
                    ori_std_deg=0.5,
                    step_noise_pos_std=0.0 if self.GOT_STUCK else 0.010,
                    step_noise_ori_std_deg=0.0 if self.GOT_STUCK else 5.0,
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.002, ori_std_deg=0.5)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.015, ori_error_threshold=0.3, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "reach_table_top_z":
            self.set_speed(max_delta_z=0.015)
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
            if self.non_markovian:
                # NM: reach_table_top_z only descends to the handoff depth; the "insert" state finishes insertion.
                target_leg_tip_pose_robot[2, 3] = self._INSERT_HANDOFF_Z - 0.002
            rel = rel_rot_mat(leg_tip_pose_robot, target_leg_tip_pose_robot)
            clean_target = rel @ ee_pose
            # NM descent pause: randomly freeze z-target to mimic human pausing to check alignment.
            # Only fires when the leg is safely above the hole (leg_tip_z_rel > STUCK_Z_HIGH).
            if self.non_markovian:
                STUCK_Z_HIGH = 0.01675
                if self.nm_insertion_pause_step_end > 0 and self.curr_cnt < self.nm_insertion_pause_step_end:
                    # Active pause: hold EE at the frozen z so the robot stops descending.
                    clean_target[2, 3] = self.nm_insertion_pause_cached_z
                else:
                    self.nm_insertion_pause_step_end = 0
                    if (
                        self.leg_tip_z_rel > STUCK_Z_HIGH
                        and self.leg_tip_z_rel < STUCK_Z_HIGH + 0.007
                        and np.random.random() < self._NM_INSERTION_PAUSE_PROB
                    ):
                        # Cache the CURRENT EE z (not the fixed destination z) so the robot holds in place.
                        self.nm_insertion_pause_cached_z = ee_pose[2, 3].item()
                        self.nm_insertion_pause_step_end = self.curr_cnt + np.random.randint(
                            self._NM_INSERTION_PAUSE_STEPS_MIN, self._NM_INSERTION_PAUSE_STEPS_MAX + 1
                        )
                        clean_target[2, 3] = self.nm_insertion_pause_cached_z
                        print(
                            f"[reach_table_top_z NM] pause started for {self.nm_insertion_pause_step_end - self.curr_cnt} steps at z={self.nm_insertion_pause_cached_z:.4f}"
                        )
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
                        step_noise_pos_std=0.002 if self.leg_tip_z_rel > 0.03 else 0.0,
                        step_noise_pos_std_z=0.002 if self.leg_tip_z_rel > 0.04 else 0.0,
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
            # NOTE: DO NOT _apply_latent_offset() here, since we need precision during insertion
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.007, ori_error_threshold=0.15, max_len=75)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "insert":
            self.set_speed(max_delta_z=0.015)
            target_leg_tip_pose_robot = torch.tensor(
                [  # Target for leg TIP: same geometry as reach_table_top_z
                    [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3] + self._LEG_HOLE_OFFSET_X],
                    [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3] + self._LEG_HOLE_OFFSET_Y],
                    [0.0, 1.0, 0.0, table_pose[2, 3] + 0.05 - self._LEG_TIP_OFFSET],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            _c, _s = np.cos(self._INSERT_TIP_RY_ANGLE), np.sin(self._INSERT_TIP_RY_ANGLE)
            ry3 = torch.tensor(
                [[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]],
                dtype=torch.float32,
                device=device,
            )
            target_leg_tip_pose_robot[:3, :3] = ry3 @ target_leg_tip_pose_robot[:3, :3]
            rel = rel_rot_mat(leg_tip_pose_robot, target_leg_tip_pose_robot)
            clean_target = rel @ ee_pose
            # Zero noise: precise final insertion, no VT walk, no latent offset
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
            target_pos[2] = 0.055 if not self.non_markovian else 0.061  # A little higher if there is target step noise

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
                    clean_target, pos_std=0.0, ori_std_deg=0.0, step_noise_pos_std=0.0025, step_noise_ori_std_deg=1.0
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.0, ori_std_deg=0.0)
            result = self.satisfy(ee_pose, target, max_len=300)
            if result == "TIMEOUT":
                timeout_failure = True
        elif state == "screw_grasp":
            # IMPORTANT: define target relative to leg pose so that we grab it centered
            target_pos = (april_to_robot @ leg_pose)[:3, 3].clone()
            target_pos[2] = 0.055
            target_ori = C.rot_mat_tensor(np.pi, 0, np.pi, device)[:3, :3]

            if self.non_markovian:
                # Alignment phase: 1..MAX_ALIGN_CYCLES random target shifts followed by one
                # clean cycle, then close the gripper.  Mimics a human repositioning before grasping.
                def new_cycle(random_offset):
                    xy = (
                        np.random.normal(0, self._NM_SCREW_GRASP_ALIGN_POS_STD, size=2) if random_offset else (0.0, 0.0)
                    )
                    self.nm_screw_grasp_align_offset = torch.tensor([*xy, 0.0], dtype=target_pos.dtype, device=device)
                    self.nm_screw_grasp_align_step_end = self.curr_cnt + np.random.randint(
                        self._NM_SCREW_GRASP_ALIGN_STEPS_MIN, self._NM_SCREW_GRASP_ALIGN_STEPS_MAX + 1
                    )

                if self.nm_screw_grasp_align_cycles_remaining is None:
                    self.nm_screw_grasp_align_cycles_remaining = np.random.randint(
                        self._NM_SCREW_GRASP_MIN_ALIGN_CYCLES, self._NM_SCREW_GRASP_MAX_ALIGN_CYCLES + 1
                    )
                    new_cycle(random_offset=True)
                    # Same +1 correction as pick_leg: curr_cnt increments at end of the
                    # transition step, so shift step_end to give the full cycle duration after the pause.
                    self.nm_screw_grasp_align_step_end += 1

                if not self.nm_screw_grasp_align_done and self.curr_cnt >= self.nm_screw_grasp_align_step_end:
                    self.nm_screw_grasp_align_cycles_remaining -= 1
                    if self.nm_screw_grasp_align_cycles_remaining > 0:
                        new_cycle(random_offset=True)
                    elif self.nm_screw_grasp_align_cycles_remaining == 0:
                        new_cycle(random_offset=False)  # final clean cycle
                        self.nm_screw_grasp_align_step_end += self._NM_SCREW_GRASP_ALIGN_CLEAN_CYCLE_EXTRA_STEPS
                    else:
                        self.nm_screw_grasp_align_done = True

                pos = target_pos if self.nm_screw_grasp_align_done else target_pos + self.nm_screw_grasp_align_offset
                clean_target = C.to_homogeneous(pos, target_ori)
                clean_target = self._apply_latent_offset(state, clean_target)
                target = self._add_noise_to_target(clean_target, pos_std=0.0, ori_std_deg=0.0)
                self.gripper_action = 1 if self.nm_screw_grasp_align_done else -1
                if self.nm_screw_grasp_align_done:
                    result = self.gripper_less(gripper_width, 2 * self.half_width + 0.001)
                    if result == "TIMEOUT":
                        timeout_failure = True
                else:
                    print(
                        f"[screw_grasp NM align] cycles_left={self.nm_screw_grasp_align_cycles_remaining}"
                        f"  offset_xy=({self.nm_screw_grasp_align_offset[0]:.4f}, {self.nm_screw_grasp_align_offset[1]:.4f})"
                    )
            else:
                clean_target = C.to_homogeneous(target_pos, target_ori)
                clean_target = self._apply_latent_offset(state, clean_target)
                target = self._add_noise_to_target(clean_target, pos_std=0.0, ori_std_deg=0.0)  # NO NOISE during grasp
                self.gripper_action = 1
                result = self.gripper_less(gripper_width, 2 * self.half_width + 0.001)
                if result == "TIMEOUT":
                    timeout_failure = True
        elif state == "screw":
            self.set_speed(delta_pos_gain=4.0)
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
