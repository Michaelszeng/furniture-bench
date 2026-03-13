import copy
import math
import pdb
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import torch
from rich import print

import furniture_bench.controllers.control_utils as C
import furniture_bench.utils.transform as T
from furniture_bench.furniture.parts.pose_filter import PoseFilter
from furniture_bench.utils.pose import get_mat, is_similar_pos, is_similar_pose, is_similar_rot, rot_mat


class Part(ABC):
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
        if elapsed >= max_len:
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
        self.curr_cnt += 1
        self._last_state = new_state
        # Only fire skill_complete on the step we first enter a skill_complete state
        return 1 if (new_state != prev_state and new_state in self.skill_complete_next_states) else 0

    def _add_noise(self, target, pos_std=0.004, ori_std_deg=4.0):
        """Apply per-step Gaussian noise to a homogeneous target matrix."""
        if self.state_no_noise():
            return target
        noisy = target.clone()
        noisy[:3, 3] += torch.normal(mean=torch.zeros(3), std=torch.tensor(pos_std, dtype=torch.float32)).to(
            target.device
        )
        ori = C.mat2quat(noisy[:3, :3]).to(target.device)
        ori = C.quat_multiply(
            ori,
            torch.tensor(
                T.axisangle2quat(
                    [
                        np.radians(np.random.normal(0, ori_std_deg)),
                        np.radians(np.random.normal(0, ori_std_deg)),
                        np.radians(np.random.normal(0, ori_std_deg)),
                    ]
                ),
                device=target.device,
            ),
        ).to(target.device)
        noisy[:3, :3] = C.quat2mat(ori)
        return noisy

    def reset(self):
        self.pre_assemble_done = False
        self._last_state = ""
        self.gripper_action = -1
        self.prev_cnt = 0
        self.curr_cnt = 0
        # Backward-compat reset for non-Markovian parts
        self.first_setting_target = True
        self.target = None

    def state_no_noise(self):
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
        """Legacy per-state-entry noise. Kept for cabinet/lamp/round_table parts."""
        if self.state_no_noise():
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
