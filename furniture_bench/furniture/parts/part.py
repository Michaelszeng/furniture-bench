import copy
import math
import pdb
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from rich import print

import furniture_bench.controllers.control_utils as C
import furniture_bench.utils.transform as T
from furniture_bench.furniture.parts.pose_filter import PoseFilter
from furniture_bench.utils.pose import get_mat, is_similar_pos, is_similar_pose, is_similar_rot, rot_mat


class Part(ABC):
    _NM_LATENT_PLAN: bool = True  # episode-level fixed position offsets per state
    _NM_STEP_NOISE: bool = False  # per-step persistent target-noise; stds passed per call to _add_noise_to_target()

    _NM_STEP_NOISE_POST_PAUSE_PROB: float = 0.005  # probability of pause after a step-noise switch
    _NM_STEP_NOISE_SWITCH_PROB: float = 0.15  # probability of resampling the step noise each timestep

    _NM_MIN_PAUSE: int = 6  # min pause duration (steps)
    _NM_MAX_PAUSE: int = 12  # max pause duration (steps); also added to every satisfy() timeout budget
    _NM_PAUSE_EXCLUDED_STATES: frozenset = frozenset()  # states where no pause is injected

    # Virtual-target walk noise tiers (used by furniture_sim_env).
    _LOW_RANDOM_WALK_NOISE_STD_STATES: frozenset = frozenset()  # sigma scaled by 0.5
    _ZERO_RANDOM_WALK_NOISE_STD_STATES: frozenset = frozenset()  # sigma set to 0 (deterministic spring)
    _ZERO_RANDOM_WALK_NOISE_Z_STD_STATES: frozenset = frozenset()  # z-axis sigma set to 0; xy sigma unchanged

    # Default speed config; copied into _current_speed on each reset_speed() call.
    _DEFAULT_SPEED_CONFIG: dict = {
        "delta_pos_gain": 2.5,
        "delta_quat_gain": 1.0,
        "max_delta_xy": 0.13,
        "max_delta_z": 0.07,
    }

    @abstractmethod
    def __init__(self, part_config, part_idx: int):
        # Three pose filter. (Each camera has filter.)
        self.pose_filter = [PoseFilter(), PoseFilter(), PoseFilter()]
        self.part_config = copy.deepcopy(part_config)
        self.name = part_config["name"]
        self.asset_file = part_config["asset_file"]
        self.tag_ids = part_config["ids"]
        self.reset_pos = part_config["reset_pos"].copy()
        self.reset_ori = part_config.get("reset_ori").copy()
        self.center_from_anchor = None  # should be set in subclass.
        self.rel_pose_from_center = {}  # should be set in subclass.
        self.reset_gripper_width = None  # should be set in subclass.
        self.rel_pose_from_center[self.tag_ids[0]] = get_mat([0, 0, 0], [0, 0, 0])  # Anchor tag.

        self.part_idx = part_idx
        self.pre_assemble_done = True
        self.pos_error_threshold = 0.01
        self.ori_error_threshold = 0.2
        self.gripper_action = -1

        self.default_assembled_pose = part_config.get("default_assembled_pose", None)
        self.collision_margin = 0.01
        # Backward-compat fields used by non-one_leg parts (cabinet, lamp, round_table)
        self.first_setting_target = True
        self.target = None
        self.prev_cnt = 0
        self.curr_cnt = 0
        self.part_moved_skill_idx = part_config.get("part_moved_skill_idx", np.inf)
        self.part_attached_skill_idx = part_config.get("part_attached_skill_idx", np.inf)
        self.no_noise = False  # set by --no-noise: disables ALL noise including step noise
        self.no_iid_target_noise = False  # set by _ENABLE_TARGET_NOISE=False: disables only i.i.d. per-step jitter
        self.dart_amount = 1.0
        self.non_markovian = False

    def randomize_init_pose(self, from_skill=0, pos_range=[-0.05, 0.05], rot_range=45):
        self.reset_pos[from_skill][:2] = self.part_config["reset_pos"][from_skill][:2] + np.random.uniform(
            pos_range[0], pos_range[1], size=2
        )  # x, y
        self.mut_ori = rot_mat(
            [0, 0, np.random.uniform(np.radians(-rot_range), np.radians(rot_range))],
            hom=True,
        )
        self.reset_ori[from_skill] = self.mut_ori @ self.part_config["reset_ori"][from_skill]

    def randomize_init_pose_high(self, high_random_idx: int):
        self.reset_pos[0] = self.part_config["high_rand_reset_pos"][high_random_idx][0]
        self.reset_ori[0] = self.part_config["high_rand_reset_ori"][high_random_idx][0]

    def is_collision(self, part2):
        """Check if the part is collided with another part without considering rotation."""
        p_x1 = -(self.reset_x_len / 2)
        p_y1 = self.reset_y_len / 2
        p1_p1 = self.mut_ori @ np.array([p_x1, p_y1, 0, 1])

        p_x2 = self.reset_x_len / 2
        p_y2 = self.reset_y_len / 2
        p1_p2 = self.mut_ori @ np.array([p_x2, p_y2, 0, 1])

        p_x3 = -(self.reset_x_len / 2)
        p_y3 = -(self.reset_y_len / 2)
        p1_p3 = self.mut_ori @ np.array([p_x3, p_y3, 0, 1])

        p_x4 = self.reset_x_len / 2
        p_y4 = -(self.reset_y_len / 2)
        p1_p4 = self.mut_ori @ np.array([p_x4, p_y4, 0, 1])

        part2_x1 = -(part2.reset_x_len / 2)
        part2_y1 = part2.reset_y_len / 2
        p2_p1 = part2.mut_ori @ np.array([part2_x1, part2_y1, 0, 1])

        part2_x2 = part2.reset_x_len / 2
        part2_y2 = part2.reset_y_len / 2
        p2_p2 = part2.mut_ori @ np.array([part2_x2, part2_y2, 0, 1])

        part2_x3 = -(part2.reset_x_len / 2)
        part2_y3 = -(part2.reset_y_len / 2)
        p2_p3 = part2.mut_ori @ np.array([part2_x3, part2_y3, 0, 1])

        part2_x4 = part2.reset_x_len / 2
        part2_y4 = -(part2.reset_y_len / 2)
        p2_p4 = part2.mut_ori @ np.array([part2_x4, part2_y4, 0, 1])

        try:
            part1_x1 = min(p1_p1[0], p1_p2[0], p1_p3[0], p1_p4[0])
            part1_x2 = max(p1_p1[0], p1_p2[0], p1_p3[0], p1_p4[0])
            part1_y1 = min(p1_p1[1], p1_p2[1], p1_p3[1], p1_p4[1])
            part1_y2 = max(p1_p1[1], p1_p2[1], p1_p3[1], p1_p4[1])

            part1_x1 = self.reset_pos[0][0] + part1_x1
            part1_x2 = self.reset_pos[0][0] + part1_x2
            part1_y1 = self.reset_pos[0][1] + part1_y1
            part1_y2 = self.reset_pos[0][1] + part1_y2

            part2_x1 = min(p2_p1[0], p2_p2[0], p2_p3[0], p2_p4[0])
            part2_x2 = max(p2_p1[0], p2_p2[0], p2_p3[0], p2_p4[0])
            part2_y1 = min(p2_p1[1], p2_p2[1], p2_p3[1], p2_p4[1])
            part2_y2 = max(p2_p1[1], p2_p2[1], p2_p3[1], p2_p4[1])

            part2_x1 = part2.reset_pos[0][0] + part2_x1
            part2_x2 = part2.reset_pos[0][0] + part2_x2
            part2_y1 = part2.reset_pos[0][1] + part2_y1
            part2_y2 = part2.reset_pos[0][1] + part2_y2
        except:
            pdb.set_trace()

        if part1_x1 > part2_x2 + self.collision_margin or part1_x2 < part2_x1 - self.collision_margin:
            return False
        if part1_y1 > part2_y2 + self.collision_margin or part1_y2 < part2_y1 - self.collision_margin:
            return False
        return True

    def in_boundary(self, pos_lim, from_skill):
        if self.reset_pos[from_skill][0] < pos_lim[0][0] or self.reset_pos[from_skill][0] > pos_lim[0][1]:
            return False
        if self.reset_pos[from_skill][1] < pos_lim[1][0] or self.reset_pos[from_skill][1] > pos_lim[1][1]:
            return False
        return True

    def reset_pose_filters(self):
        for pose_filter in self.pose_filter:
            pose_filter.reset()

    def is_in_reset_ori(self, pose: npt.NDArray[np.float32], from_skill: int, ori_bound: float) -> bool:
        reset_ori = self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        if is_similar_rot(pose[:3, :3], reset_ori[:3, :3], ori_bound=ori_bound):
            return True
        return False

    def is_in_reset_pose(self, pose, from_skill, pos_threshold, ori_bound):
        if self.is_in_reset_pos(pose, from_skill, pos_threshold) and self.is_in_reset_ori(pose, from_skill, ori_bound):
            return True
        print(f"[reset] Part {self.__class__.__name__} [{self.part_idx}] is not in the reset pose.")

        if not self.is_in_reset_pos(pose, from_skill, pos_threshold):
            print(
                "xy should be ({0:0.3f}, {1:0.3f}), but got ({2:0.3f}, {3:0.3f})".format(
                    self.reset_pos[from_skill][0],
                    self.reset_pos[from_skill][1],
                    pose[0, 3],
                    pose[1, 3],
                )
            )
            return False
        if not self.is_in_reset_ori(pose, from_skill, ori_bound):
            print("Reset orientation mismatch.")
            return False

    def is_in_reset_pos(self, pose, from_skill, pos_threshold):
        """check whether (x, y) position is in reset position."""
        reset_pos = self.reset_pos[from_skill][:2]
        part_pos = np.array(reset_pos)
        detected_pos = np.array(pose[:2, 3])
        return is_similar_pos(part_pos[:2], detected_pos[:2], pos_threshold=pos_threshold)

    def assemble_done(self, rel_pose, assembled_rel_poses):
        for assembled_rel_pose in assembled_rel_poses:
            if is_similar_pose(
                assembled_rel_pose,
                rel_pose,
                ori_bound=0.96,
                pos_threshold=[0.005, 0.005, 0.005],
            ):
                return True
        return False

    def satisfy(
        self,
        current,
        target,
        pos_error_threshold=None,
        ori_error_threshold=None,
        max_len=25,
    ):
        """Check whether current pose satisfies target pose.

        Returns True if within thresholds, False if not yet there, "TIMEOUT" if max_len exceeded.
        """
        if pos_error_threshold is None:
            pos_error_threshold = self.pos_error_threshold
        if ori_error_threshold is None:
            ori_error_threshold = self.ori_error_threshold

        pos_err = (current[:3, 3] - target[:3, 3]).abs().sum()
        ori_err = (target[:3, :3] - current[:3, :3]).abs().sum()
        print(f"pos_err: {pos_err.item():.4f}, ori_err: {ori_err.item():.4f}")
        if pos_err < pos_error_threshold and ori_err < ori_error_threshold:
            return True
        elapsed = self.curr_cnt - self.prev_cnt
        if elapsed >= max_len + self.max_len_offset:
            print(
                f"[TIMEOUT] {self.name} satisfy after {elapsed} steps (pos_err={pos_err.item():.4f}, ori_err={ori_err.item():.4f})"
            )
            return "TIMEOUT"
        return False

    def gripper_less(self, gripper_width, target_width, cnt_max=50):
        """Check if gripper width is less than target width."""
        if gripper_width <= target_width:
            return True
        elapsed = self.curr_cnt - self.prev_cnt
        if elapsed >= cnt_max:
            print(f"[TIMEOUT] {self.name} gripper_less after {elapsed} steps (width={gripper_width.item():.4f})")
            return "TIMEOUT"
        return False

    def gripper_greater(self, gripper_width, target_width, cnt_max=30):
        """Check if gripper width is greater than target width."""
        if gripper_width >= target_width:
            return True
        elapsed = self.curr_cnt - self.prev_cnt
        if elapsed >= cnt_max:
            print(f"[TIMEOUT] {self.name} gripper_greater after {elapsed} steps (width={gripper_width.item():.4f})")
            return "TIMEOUT"
        return False

    def state_transition_handler(self, prev_state, new_state) -> int:
        """Handle state change bookkeeping: reset timeout window on transition, return skill_complete flag."""
        if new_state != prev_state:
            print(f"[yellow][FSM][/yellow] {self.name}: {prev_state} -> {new_state}")
            self.prev_cnt = self.curr_cnt
            self._nm_sn_switch_pending = False  # cancel any deferred step-noise switch
        self.curr_cnt += 1
        self._last_state = new_state
        # Only fire skill_complete on the step we first enter a skill_complete state
        return 1 if (new_state != prev_state and new_state in self.skill_complete_next_states) else 0

    def _add_noise_to_target(
        self,
        target,
        pos_std=0.004,
        pos_std_z=None,
        ori_std_deg=4.0,
        pos_max=None,
        ori_max_deg=None,
        step_noise_pos_std=0.0,
        step_noise_pos_std_z=None,
        step_noise_ori_std_deg=0.0,
        step_noise_pauses=True,
    ):
        """Add noise to TARGET.

        Step noise (persistent expert variation) is applied in-place to the input `target`
        so that callers' clean_target automatically includes it. i.i.d. per-step jitter is
        DART noise and is only applied to the returned clone.

        Args:
            pos_std / pos_std_z / ori_std_deg: i.i.d. per-step DART noise stds (position in m, orientation in deg).
            pos_max / ori_max_deg: clip the i.i.d. noise to [-x, x].
            step_noise_pos_std / step_noise_pos_std_z / step_noise_ori_std_deg: persistent expert step-noise stds;
                NOT scaled by dart_amount. Default 0 = no step noise.
        """
        if self.no_noise:
            return target

        device = target.device

        # NM step noise: persistent expert offset applied in-place to `target` (= clean_target in callers).
        # Not scaled by dart_amount — this noise is part of the expert and is recorded in clean actions.
        if self.non_markovian and self._NM_STEP_NOISE and (step_noise_pos_std > 0.0 or step_noise_ori_std_deg > 0.0):
            state = self._last_state
            wants_switch = state not in self._step_noise or np.random.random() < self._NM_STEP_NOISE_SWITCH_PROB

            if wants_switch:
                self._perform_step_noise_switch(
                    state,
                    step_noise_pos_std,
                    step_noise_pos_std_z,
                    step_noise_ori_std_deg,
                    enable_post_pause=step_noise_pauses,
                )

            if state in self._step_noise:
                step = self._step_noise[state]
                target[:3, 3] += torch.tensor(step["pos"], dtype=target.dtype, device=device)
                if step["ori"].any():
                    ori = C.mat2quat(target[:3, :3]).to(device)
                    ori = C.quat_multiply(
                        ori,
                        torch.tensor(T.axisangle2quat(step["ori"].tolist()), device=device),
                    ).to(device)
                    target[:3, :3] = C.quat2mat(ori)

        # Clone after step noise; i.i.d. DART noise is applied only to the clone (not recorded in clean actions).
        noisy = target.clone()

        if not self.no_iid_target_noise and not self.current_state_no_noise():
            scaled_pos_std = pos_std * self.dart_amount
            scaled_pos_std_z = (pos_std_z if pos_std_z is not None else pos_std) * self.dart_amount
            iid_pos = torch.normal(
                mean=torch.zeros(3),
                std=torch.tensor([scaled_pos_std, scaled_pos_std, scaled_pos_std_z], dtype=torch.float32),
            ).to(device)
            if pos_max is not None:
                iid_pos = iid_pos.clamp(-pos_max, pos_max)
            noisy[:3, 3] += iid_pos

            scaled_ori_std_rad = np.radians(ori_std_deg * self.dart_amount)
            iid_ori = np.random.normal(0, scaled_ori_std_rad, size=3)
            if ori_max_deg is not None:
                iid_ori = np.clip(iid_ori, -np.radians(ori_max_deg), np.radians(ori_max_deg))
            if iid_ori.any():
                ori = C.mat2quat(noisy[:3, :3]).to(device)
                ori = C.quat_multiply(
                    ori,
                    torch.tensor(T.axisangle2quat(iid_ori.tolist()), device=device),
                ).to(device)
                noisy[:3, :3] = C.quat2mat(ori)

        return noisy

    def _perform_step_noise_switch(self, state, pos_std, pos_std_z, ori_std_deg, enable_post_pause: bool = True):
        """Sample new step noise for `state` and optionally arm a post-switch pause."""
        sp = pos_std
        sz = pos_std_z if pos_std_z is not None else pos_std
        so = np.radians(ori_std_deg)
        self._step_noise[state] = {
            "pos": np.random.normal(0, [sp, sp, sz]),
            "ori": np.random.normal(0, so, size=3),
        }
        if enable_post_pause and np.random.random() < self._NM_STEP_NOISE_POST_PAUSE_PROB:
            self._nm_sn_post_pause_pending = True
            self._nm_sn_post_pause_steps = int(np.random.randint(self._NM_MIN_PAUSE // 2, self._NM_MAX_PAUSE // 2 + 1))

    def _nm_sticky_delay(self, from_state: str, min_delay: int, max_delay: int) -> bool:
        """Return True if a sticky transition delay is still active (caller should stay in from_state).

        On the first call for a given from_state, samples a uniform countdown in [0, max_delay].
        Decrements each call until zero, then clears the entry and returns False so the transition fires.
        Passing max_delay=0 makes the transition immediate (no stickiness).
        """
        if from_state not in self._nm_sticky_countdowns:
            self._nm_sticky_countdowns[from_state] = int(np.random.randint(min_delay, max_delay + 1))
        if self._nm_sticky_countdowns[from_state] > 0:
            self._nm_sticky_countdowns[from_state] -= 1
            return True
        del self._nm_sticky_countdowns[from_state]
        return False

    def _nm_defer_transition_with_pause(self, next_state: str) -> str:
        """Arm a pre-entry pause and hold in the current state until it completes.

        Sets _nm_pending_next_state and requests a pause from the env, then returns
        _last_state so the caller's compute method keeps executing the current state.
        On the next FSM call after the pause, _nm_pop_pending_transition() fires the transition.
        """
        if not self._NM_PAUSES:
            return next_state
        self._nm_pending_next_state = next_state
        self._nm_transition_pause_requested = True
        return self._last_state

    def _nm_pop_pending_transition(self) -> Optional[str]:
        """Return and clear the deferred next state, or None if no transition is pending."""
        if self._nm_pending_next_state is not None:
            result = self._nm_pending_next_state
            self._nm_pending_next_state = None
            return result
        return None

    def reset_speed(self):
        """Restore _current_speed to defaults; called by the env before each FSM step."""
        self._current_speed = self._DEFAULT_SPEED_CONFIG.copy()

    def set_speed(self, delta_pos_gain=None, delta_quat_gain=None, max_delta_xy=None, max_delta_z=None):
        """Override one or more speed parameters for the current step.

        Call this anywhere inside fsm_step / pre_assemble to change EE speed.
        The env resets _current_speed to defaults before each step, so each state
        starts fresh and can call set_speed() as many times as needed.
        """
        if delta_pos_gain is not None:
            self._current_speed["delta_pos_gain"] = delta_pos_gain
        if delta_quat_gain is not None:
            self._current_speed["delta_quat_gain"] = delta_quat_gain
        if max_delta_xy is not None:
            self._current_speed["max_delta_xy"] = max_delta_xy
        if max_delta_z is not None:
            self._current_speed["max_delta_z"] = max_delta_z

    def get_speed_config(self) -> dict:
        """Return the speed config set for this step (may have been modified by set_speed())."""
        return self._current_speed

    def reset(self):
        self.pre_assemble_done = False
        self._last_state = ""
        self.gripper_action = -1
        self.prev_cnt = 0
        self.curr_cnt = 0
        self.max_len_offset = 0  # extra steps added to every satisfy() budget; set to _NM_MAX_PAUSE for non-Markovian
        self._current_speed = self._DEFAULT_SPEED_CONFIG.copy()
        # Non-Markovian latent offset (populated by apply_non_markovian_config)
        self.latent_offsets = {}
        self._step_noise = {}
        self._nm_sn_post_pause_pending = False  # set by _perform_step_noise_switch; read by env to arm a pause
        self._nm_sn_post_pause_steps = 0
        # Sticky-transition countdowns: key = from_state, value = steps remaining before transition fires.
        self._nm_sticky_countdowns: dict[str, int] = {}
        # Set True in compute_state/compute_pre_assemble_state before returning a state that should trigger a pause.
        # Consumed (reset to False) by the env after arming the pause.
        self._nm_transition_pause_requested: bool = False
        # Deferred transition: next state to enter once the pause completes (None = no pending transition).
        self._nm_pending_next_state: Optional[str] = None
        # Backward-compat reset for non-Markovian parts
        self.first_setting_target = True
        self.target = None

    def _lo(self, state: str = None) -> np.ndarray:
        """Return the 3-element position latent offset for `state` (default: _last_state).
        Returns zeros when _NM_LATENT_PLAN is off or the state has no entry."""
        if not self._NM_LATENT_PLAN:
            return np.zeros(3)
        s = state if state is not None else self._last_state
        entry = self.latent_offsets.get(s)
        return entry["pos"] if entry is not None else np.zeros(3)

    def _lo_ori(self, state: str = None) -> np.ndarray:
        """Return the 3-element axis-angle orientation latent offset for `state` (default: _last_state).
        Returns zeros when _NM_LATENT_PLAN is off or the state has no entry."""
        if not self._NM_LATENT_PLAN:
            return np.zeros(3)
        s = state if state is not None else self._last_state
        entry = self.latent_offsets.get(s)
        return entry["ori"] if entry is not None else np.zeros(3)

    def _apply_latent_offset(self, state: str, clean_target):
        """Apply the episode-level latent plan offset (position + orientation) to clean_target.
        No-op when latent_offsets is empty (non-Markovian disabled or _NM_LATENT_PLAN=False).
        """
        if self.latent_offsets:
            offset = self.latent_offsets.get(state)
            if offset is not None:
                clean_target = clean_target.clone()
                device = clean_target.device
                clean_target[:3, 3] += torch.tensor(offset["pos"], device=device, dtype=clean_target.dtype)
                ori_vec = offset["ori"]
                if np.any(ori_vec):
                    quat = C.mat2quat(clean_target[:3, :3])
                    delta = torch.tensor(T.axisangle2quat(ori_vec.tolist()), device=device, dtype=clean_target.dtype)
                    clean_target[:3, :3] = C.quat2mat(C.quat_multiply(quat, delta))
        return clean_target

    def current_state_no_noise(self):
        return False

    def current_state_low_action_noise(self):
        return False

    def current_state_clean_action_noise(self):
        """Return True if the clean_action should also have noise added in this state."""
        return False

    # ---- Backward-compat methods for non-one_leg parts (cabinet, lamp, round_table) ----

    def may_transit_state(self, next_state):
        """Legacy non-Markovian state transition. Kept for cabinet/lamp/round_table parts."""
        skill_complete = 0
        if next_state != self._state:
            print(f"Changing state from {self._state} to {next_state}")
            self._state = next_state
            if next_state in self.skill_complete_next_states:
                skill_complete = 1
            self.first_setting_target = True
            self.prev_cnt = self.curr_cnt
        self.curr_cnt += 1
        return skill_complete

    def add_noise_first_target(self, target, pos_noise=None, ori_noise=None):
        """
        Legacy per-state-entry noise. Kept for cabinet/lamp/round_table parts.
        Adds a fixed noise to the target pose on the first time entering a state (rather than adding random target
        noise each timestep).
        TODO: REMOVE THIS.
        """
        if self.no_noise or self.no_iid_target_noise or self.current_state_no_noise():
            return target
        if self.first_setting_target:
            if pos_noise is not None:
                target[:3, 3] += pos_noise
            else:
                target[:3, 3] += torch.normal(mean=torch.zeros((3,)), std=torch.ones((3,)) * 0.003).to(target.device)
            ori = C.mat2quat(target[:3, :3]).to(target.device)
            if ori_noise is not None:
                ori = C.quat_multiply(ori, ori_noise).to(target.device)
            else:
                ori = C.quat_multiply(
                    ori,
                    torch.tensor(
                        T.axisangle2quat(
                            [
                                np.radians(np.random.normal(0, 3)),
                                np.radians(np.random.normal(0, 3)),
                                np.radians(np.random.normal(0, 3)),
                            ]
                        ),
                        device=target.device,
                    ),
                ).to(target.device)
            self.target = C.to_homogeneous(target[:3, 3], C.quat2mat(ori))
            self.first_setting_target = False
        return self.target
