from re import L

import numpy as np
import numpy.typing as npt
import torch
from numpy.linalg import inv
from rich import print

import furniture_bench.controllers.control_utils as C
import furniture_bench.utils.transform as T
from furniture_bench.config import config
from furniture_bench.furniture.parts.part import Part
from furniture_bench.utils.pose import get_mat, is_similar_rot, rot_mat


class TableTop(Part):
    # ── Per-state noise tiers ──────────────────────────────────────────────────
    # Target-noise tiers (used by apply_non_markovian_config for latent offsets / step noise stds).
    # NOTE: these do not affect the random per-step target noise in _add_noise_to_target(), only affect the latent
    # plan offsets and step noise stds.
    _LOW_LATENT_TARGET_STD_STATES: frozenset = frozenset(
        {"reach_body_grasp_z", "pick_body", "release", "go_up", "done"}
    )
    _ZERO_LATENT_TARGET_STD_STATES: frozenset = frozenset({"push"})

    # Virtual-target walk noise tiers (used by furniture_sim_env).
    _LOW_RANDOM_WALK_NOISE_STD_STATES: frozenset = frozenset({"reach_body_grasp_z", "pick_body", "push", "release"})
    _ZERO_RANDOM_WALK_NOISE_STD_STATES: frozenset = frozenset()

    # ── Sticky transition delays ───────────────────────────────────────────
    # Min/Max extra steps the robot lingers in "push" after at_push_xy is first satisfied,
    # before transitioning to "release".
    _NM_STICKY_PUSH_RELEASE_MAX_DELAY: int = 16
    _NM_STICKY_PUSH_RELEASE_MIN_DELAY: int = 9

    # Min/Max extra steps the robot lingers in "reach_body_grasp_z" after at_body_z is first satisfied,
    # before transitioning to "pick_body".
    _NM_STICKY_REACH_BODY_GRASP_Z_PICK_BODY_MAX_DELAY: int = 12
    _NM_STICKY_REACH_BODY_GRASP_Z_PICK_BODY_MIN_DELAY: int = 7

    # ── Desired table-top body XY position ────────────────────────────────────
    # Desired table-top body XY position when correctly seated against the obstacles.
    # Used by the NM at_push_xy condition.  Calibrate empirically.
    _NM_PUSH_BODY_TARGET_X: float = 0.5902
    _NM_PUSH_BODY_TARGET_Y: float = 0.0838

    # ── NM push overshoot: shift push target in +x/+y as body nears goal ──────
    # Once the body is within THRESHOLD of its target XY, the push target gains an
    # extra offset in +x and +y equal to SCALE / (remaining_dist + C).  This makes
    # the EE push further as the body converges, preventing it from stalling short.
    _NM_PUSH_OVERSHOOT_THRESHOLD: float = 0.011  # activation distance (m)
    _NM_PUSH_OVERSHOOT_SCALE: float = 0.01  # numerator of 1/d term (m²)
    _NM_PUSH_OVERSHOOT_C: float = 0.001  # denominator constant to prevent divergence (m)

    def __init__(self, part_config: dict, part_idx: int):
        super().__init__(part_config, part_idx)

        self.gripper_action = -1
        self.body_grip_width = 0.01
        # Fractional offset along the grasped side: 0.0 = center, 0.5 = 3/4 from one end.
        self.grasp_side_offset_frac = -0.18 if self.non_markovian else -0.23  # Closer to center for NM

        self.skill_complete_next_states = [
            "push",
            "go_up",
        ]  # Specificy next state after skill is complete.
        self.reset()

    def reset(self):
        super().reset()  # resets prev_cnt=0, curr_cnt=0, pre_assemble_done, etc.
        self._last_state = "reach_body_grasp_xy"
        self.gripper_action = -1
        self._grasp_side_offset_robot = None  # set once grasp target is first computed

    def apply_non_markovian_config(self):
        """Sample episode-level non-Markovian latent variables for table-top pre-assembly."""
        if not self._NM_LATENT_PLAN:
            return

        HIGH_STD = 0.007
        LOW_STD = 0.003
        HIGH_ORI_STD = np.radians(3.0)
        LOW_ORI_STD = np.radians(2.0)

        all_states = [
            "reach_body_grasp_xy",
            "reach_body_grasp_z",
            "pick_body",
            "push",
            "release",
            "go_up",
            "done",
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
        offsets_str = "\n".join(
            f"  {state}: pos={v['pos']}, ori={v['ori']}" for state, v in self.latent_offsets.items()
        )
        print(f"[TABLE_TOP] latent_offsets:\n{offsets_str}")

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        reset_ori = self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        for _ in range(4):
            if is_similar_rot(pose[:3, :3], reset_ori[:3, :3], ori_bound=ori_bound):
                return True
            pose = pose @ rot_mat(np.array([0, np.pi / 2, 0]), hom=True)
        return False

    def _find_closest_y(self, pose):
        closest_y = pose.clone()
        for i in range(4):
            tmp_pose = pose @ torch.tensor(self.rel_pose_from_center[self.tag_ids[i]]).float().to(pose.device)
            if tmp_pose[1, 3] < closest_y[1, 3]:
                closest_y = tmp_pose
        return closest_y

    def _get_body_pose_robot(self, rb_states, part_idxs, sim_to_april_mat, april_to_robot):
        """Extract the table-top body pose in robot frame."""
        body_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        body_pose = sim_to_april_mat @ body_pose
        return april_to_robot @ body_pose, body_pose  # (robot frame, april frame)

    def _get_grasp_target(self, body_pose_april, april_to_robot, ee_pos, device):
        """Compute the grasp target pose for reach_body_grasp_xy."""
        body_pose = self._find_closest_y(body_pose_april)
        rot = body_pose[:4, :4] @ torch.tensor(rot_mat([np.pi / 2, 0, 0], hom=True), device=device)
        pos = body_pose[:3, 3]
        pos = torch.concat([pos, torch.tensor([1.0], device=device)])
        target_pos = (april_to_robot @ pos)[:3]
        target_ori = (april_to_robot @ rot)[:3, :3]
        if self.non_markovian:
            target_pos[2] = body_pose_april[2, 3] + 0.035  # Slightly above the table top to avoid collision
        else:
            target_pos[2] = ee_pos[2]  # keep current Z
        # Shift grasp from the center of the side to 3/4 along it.
        # body_pose local x-axis runs along the face; half_width/2 = 1/4 of the full
        # side length, moving the grasp point from 1/2 to 3/4 from one end.
        side_dir_april = body_pose[:3, 0]
        side_dir_robot = april_to_robot[:3, :3] @ side_dir_april
        grasp_offset = side_dir_robot * (self.half_width * self.grasp_side_offset_frac)
        self._grasp_side_offset_robot = grasp_offset
        target_pos = target_pos + grasp_offset
        return C.to_homogeneous(target_pos, target_ori)

    def _get_push_target(
        self,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        ee_pose,
        device,
        body_pose_robot,
        overshoot: bool = False,
    ):
        """Compute the push target position from obstacle poses."""
        target_pos = torch.zeros((4,), device=device)
        target_pos[-1] = 1
        for name in ["obstacle_front", "obstacle_right", "obstacle_left"]:
            obstacle_pos = torch.cat(
                [
                    rb_states[part_idxs[name]][0][:3],
                    torch.tensor([1.0], device=device),
                ]
            )
            target_pos[0] = max(obstacle_pos[0], target_pos[0])
            target_pos[1] = max(obstacle_pos[1], target_pos[1])
        target_pos = april_to_robot @ sim_to_april_mat @ target_pos
        target_pos[0] -= self.half_width * 2
        target_pos[0] -= 0.005  # Empirical offset to avoid early collision with the long edge of theobstacle
        target_pos[1] -= self.half_width
        target_pos[2] = body_pose_robot[2, 3]
        target_pos = target_pos[:3]
        if self._grasp_side_offset_robot is not None:
            target_pos = target_pos + self._grasp_side_offset_robot
        if self.non_markovian and overshoot:
            # NM push overshoot: shift push target in +x/+y as body nears goal
            body_xy = body_pose_robot[:2, 3]
            goal_xy = torch.tensor(
                [self._NM_PUSH_BODY_TARGET_X, self._NM_PUSH_BODY_TARGET_Y],
                dtype=body_xy.dtype,
                device=device,
            )
            remaining_dist = float((body_xy - goal_xy).norm())
            if remaining_dist < self._NM_PUSH_OVERSHOOT_THRESHOLD:
                # https://www.desmos.com/calculator/kglflxsmhg
                shift = self._NM_PUSH_OVERSHOOT_SCALE / (remaining_dist + self._NM_PUSH_OVERSHOOT_C) - (
                    self._NM_PUSH_OVERSHOOT_SCALE / (self._NM_PUSH_OVERSHOOT_THRESHOLD + self._NM_PUSH_OVERSHOOT_C)
                )
                target_pos[0] = target_pos[0] + shift
                # target_pos[1] = target_pos[1] + shift
        target_ori = torch.zeros((3, 3), device=device)
        target_ori[0][1] = 1
        target_ori[1][0] = 1
        target_ori[2][2] = -1
        return C.to_homogeneous(target_pos, target_ori)

    def compute_pre_assemble_state(
        self,
        ee_pos,
        ee_quat,
        gripper_width,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
    ) -> str:
        """Determine pre-assembly FSM state from current environment state."""
        device = ee_pos.device
        ################################################################################################################
        # GET CURRENT ROBOT AND ENVIRONMENT STATE
        ################################################################################################################
        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
        body_pose_robot, body_pose_april = self._get_body_pose_robot(
            rb_states, part_idxs, sim_to_april_mat, april_to_robot
        )

        ################################################################################################################
        # SHARED TARGETS AND CONDITIONS
        # lo() returns zeros when non_markovian=False (latent_offsets is never populated),
        # so these booleans are correct for both the NM and Markovian paths.
        ################################################################################################################
        grasp_target = self._get_grasp_target(body_pose_april, april_to_robot, ee_pos, device)
        push_target = self._get_push_target(
            rb_states,
            part_idxs,
            sim_to_april_mat,
            april_to_robot,
            ee_pose,
            device,
            body_pose_robot,
        )

        # Latent offsets for the current state used by Non-Markovian policy
        lo_t = torch.tensor(self._lo(), dtype=ee_pos.dtype, device=device)  # zeros when non-NM
        lo_xy, lo_z = lo_t[:2], float(lo_t[2])

        gripper_open_thr = config["robot"]["max_gripper_width"]["square_table"] - 0.005
        gripper_closed_thr = self.body_grip_width + 0.005

        gripper_open = gripper_width >= gripper_open_thr
        gripper_closed = gripper_width <= gripper_closed_thr
        if self.non_markovian:
            at_grasp_xy = (ee_pos[:2] - (grasp_target[:2, 3] + lo_xy)).abs().sum() < self.pos_error_threshold * 5
        else:
            at_grasp_xy = (ee_pos[:2] - (grasp_target[:2, 3] + lo_xy)).abs().sum() < self.pos_error_threshold * 4
        self.grasp_z_offset = 0.009 if self.non_markovian else 0.0
        at_body_z = abs(ee_pos[2] - (body_pose_robot[2, 3] + self.grasp_z_offset + lo_z)) < self.pos_error_threshold * 2
        if self.non_markovian:
            body_target_xy = torch.tensor(
                [self._NM_PUSH_BODY_TARGET_X, self._NM_PUSH_BODY_TARGET_Y],
                dtype=ee_pos.dtype,
                device=device,
            )
            at_push_xy = (
                body_pose_robot[:2, 3] - body_target_xy
            ).abs().sum() < self.pos_error_threshold  # Smaller threshold for NM
        else:
            at_push_xy = (ee_pos[:2] - (push_target[:2, 3] + lo_xy)).abs().sum() < self.pos_error_threshold * 3
        z_high = ee_pos[2] >= 0.09 + lo_z
        # Whether the table body has been physically pushed near the push target.
        # This prevents "done"/"go_up" from firing at episode start when the EE happens
        # to start near push_xy (gripper open, EE high) before any push has occurred.
        body_near_push = (body_pose_robot[:2, 3] - push_target[:2, 3]).abs().sum() < 0.1

        ################################################################################################################
        # NON-MARKOVIAN: sequential — only check whether _last_state has been completed
        ################################################################################################################
        if self.non_markovian:
            state_sequence = [
                "reach_body_grasp_xy",
                "reach_body_grasp_z",
                "pick_body",
                "push",
                "release",
                "go_up",
                "done",
            ]
            current = self._last_state
            nxt = state_sequence[state_sequence.index(current) + 1] if current != "done" else "done"

            # Deferred transition: the pause just completed — fire the pending state change now.
            pending = self._nm_pop_pending_transition()
            if pending is not None:
                return pending

            if current == "reach_body_grasp_xy" and at_grasp_xy:
                # return self.c(nxt)
                return nxt
            elif current == "reach_body_grasp_z" and at_body_z:
                if self._nm_sticky_delay(
                    "reach_body_grasp_z",
                    self._NM_STICKY_REACH_BODY_GRASP_Z_PICK_BODY_MIN_DELAY,
                    self._NM_STICKY_REACH_BODY_GRASP_Z_PICK_BODY_MAX_DELAY,
                ):
                    return current  # linger in "reach_body_grasp_z" before transitioning to "pick_body"
                # return self._nm_defer_transition_with_pause(nxt)
                return nxt
            elif current == "pick_body" and gripper_closed:
                # return self._nm_defer_transition_with_pause(nxt)
                return nxt
            elif current == "push" and at_push_xy:
                if self._nm_sticky_delay(
                    "push", self._NM_STICKY_PUSH_RELEASE_MIN_DELAY, self._NM_STICKY_PUSH_RELEASE_MAX_DELAY
                ):
                    return current  # linger in "push" before releasing
                # return self._nm_defer_transition_with_pause(nxt)
                return nxt
            elif current == "release" and gripper_open:
                # return self._nm_defer_transition_with_pause(nxt)
                return nxt
            elif current == "go_up" and z_high:
                # return self._nm_defer_transition_with_pause(nxt)
                return nxt
            return current

        ################################################################################################################
        # MARKOVIAN: hierarchical — FSM state determined entirely from current environment state
        ################################################################################################################
        if at_push_xy:
            if gripper_open:
                # Gripper open at push target: done/go_up only if table was actually pushed.
                if body_near_push:
                    if z_high:
                        return "done"
                    return "go_up"
                # Gripper open but table not pushed yet → episode-start false positive;
                # fall through to the grasp/pick states below.
            else:
                # Gripper not fully open at push target → open gripper to release table.
                return "release"

        # "push": gripper fully closed — table body is grasped, move to push target.
        if gripper_closed:
            return "push"
        # "pick_body": EE at body Z, gripper closing but not closed yet
        if at_body_z:
            return "pick_body"
        # "reach_body_grasp_z": EE at body XY, needs to descend to body Z
        if at_grasp_xy:
            return "reach_body_grasp_z"
        return "reach_body_grasp_xy"

    def pre_assemble(
        self,
        ee_pos,
        ee_quat,
        gripper_width,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
    ):
        state = self.compute_pre_assemble_state(
            ee_pos, ee_quat, gripper_width, rb_states, part_idxs, sim_to_april_mat, april_to_robot
        )

        # Reset timeout window as soon as a new state is entered, before action code runs.
        if state != self._last_state:
            self.prev_cnt = self.curr_cnt

        # Throttled state print (every 10 steps)
        if self.curr_cnt % 10 == 0:
            print(f"[magenta][TABLE_TOP][/magenta] pre_assemble state={state} (step={self.curr_cnt})")

        timeout_failure = False

        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
        body_pose_robot, body_pose_april = self._get_body_pose_robot(
            rb_states, part_idxs, sim_to_april_mat, april_to_robot
        )
        body_pose_robot[2, 3] += 0.01  # Make robot grasp table_top slightly higher
        device = ee_pose.device

        # Default target: stay at current EE pose
        target = ee_pose.clone()
        clean_target = ee_pose.clone()

        if state == "reach_body_grasp_xy":
            self.gripper_action = -1  # approaching body, gripper open
            clean_target = self._get_grasp_target(body_pose_april, april_to_robot, ee_pos, device)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(clean_target, step_noise_pos_std=0.020, step_noise_ori_std_deg=5.0)
            else:
                target = self._add_noise_to_target(clean_target)
            result = self.satisfy(ee_pose, target, max_len=300)
            if result == "TIMEOUT":
                timeout_failure = True

        elif state == "reach_body_grasp_z":
            self.gripper_action = -1  # descending to grasp body, gripper still open
            grasp_target = self._get_grasp_target(body_pose_april, april_to_robot, ee_pos, device)
            target_pos = grasp_target[:3, 3].clone()
            target_pos[2] = body_pose_robot[2, 3] + self.grasp_z_offset
            target_ori = grasp_target[:3, :3]

            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target,
                    pos_std=0.005,  # slightly larger noise for Z approach (original: [0.01, 0.01, 0.001])
                    step_noise_pos_std=0.004,
                    step_noise_ori_std_deg=3.0,
                )
            else:
                target = self._add_noise_to_target(
                    clean_target,
                    pos_std=0.005,  # slightly larger noise for Z approach (original: [0.01, 0.01, 0.001])
                )
            result = self.satisfy(ee_pose, target, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True

        elif state == "pick_body":
            grasp_target = self._get_grasp_target(body_pose_april, april_to_robot, ee_pos, device)
            target_pos = grasp_target[:3, 3].clone()
            target_pos[2] = body_pose_robot[2, 3]
            target_ori = grasp_target[:3, :3]
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target, pos_std=0.003, ori_std_deg=3.0, step_noise_pos_std=0.002, step_noise_ori_std_deg=2.0
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.003, ori_std_deg=3.0)
            self.gripper_action = 1  # grasping body, grippers closed
            result = self.gripper_less(gripper_width, self.body_grip_width)
            if result == "TIMEOUT":
                timeout_failure = True

        elif state == "push":
            self.gripper_action = 1  # body grasped, hold closed while pushing into place
            if self.non_markovian:
                self.set_speed(delta_pos_gain=5.0, max_delta_xy=0.5)
            clean_target = self._get_push_target(
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                ee_pose,
                device,
                body_pose_robot,
                overshoot=True if self.non_markovian else False,
            )
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target,
                    pos_std=0.001,
                    step_noise_pos_std=0.008,
                    step_noise_ori_std_deg=1.0,
                    pos_max=0.08,
                    ori_max_deg=1.0,
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.001)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.5, max_len=200)
            if result == "TIMEOUT":
                timeout_failure = True

        elif state == "release":
            clean_target = self._get_push_target(
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                ee_pose,
                device,
                body_pose_robot,
            )
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target, pos_std=0.003, ori_std_deg=3.0, step_noise_pos_std=0.003, step_noise_ori_std_deg=3.0
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.003, ori_std_deg=3.0)
            self.gripper_action = -1  # releasing body, gripper open
            result = self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["square_table"] - 0.001,
            )
            if result == "TIMEOUT":
                timeout_failure = True

        elif state == "go_up":
            self.gripper_action = -1  # body already released, EE retreating
            push_target = self._get_push_target(
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                ee_pose,
                device,
                body_pose_robot,
            )
            target_pos = push_target[:3, 3].clone()
            target_pos[2] = 0.1
            target_ori = push_target[:3, :3]
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(clean_target, step_noise_pos_std=0.01, step_noise_ori_std_deg=6.0)
            else:
                target = self._add_noise_to_target(clean_target)
            result = self.satisfy(ee_pose, target, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True

        elif state == "done":
            self.pre_assemble_done = True
            push_target = self._get_push_target(
                rb_states,
                part_idxs,
                sim_to_april_mat,
                april_to_robot,
                ee_pose,
                device,
                body_pose_robot,
            )
            target_pos = push_target[:3, 3].clone()
            target_pos[2] = 0.1
            target_ori = push_target[:3, :3]
            clean_target = C.to_homogeneous(target_pos, target_ori)
            clean_target = self._apply_latent_offset(state, clean_target)
            if self.non_markovian:
                target = self._add_noise_to_target(
                    clean_target, pos_std=0.003, ori_std_deg=3.0, step_noise_pos_std=0.003, step_noise_ori_std_deg=3.0
                )
            else:
                target = self._add_noise_to_target(clean_target, pos_std=0.003, ori_std_deg=3.0)
            self.gripper_action = -1  # gripper open

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
