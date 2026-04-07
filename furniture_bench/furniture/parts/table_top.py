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
    def __init__(self, part_config: dict, part_idx: int):
        super().__init__(part_config, part_idx)

        self.gripper_action = -1
        self.body_grip_width = 0.01
        # Fractional offset along the grasped side: 0.0 = center, 0.5 = 3/4 from one end.
        self.grasp_side_offset_frac = -0.25

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
        self, rb_states, part_idxs, sim_to_april_mat, april_to_robot, ee_pose, device, body_pose_robot
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
        target_pos[0] -= 0.005  # Empirical offset to avoid early collision with the obstacle
        target_pos[1] -= self.half_width
        target_pos[2] = body_pose_robot[2, 3]
        target_pos = target_pos[:3]
        if self._grasp_side_offset_robot is not None:
            target_pos = target_pos + self._grasp_side_offset_robot
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
        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
        body_pose_robot, body_pose_april = self._get_body_pose_robot(
            rb_states, part_idxs, sim_to_april_mat, april_to_robot
        )
        push_target = self._get_push_target(
            rb_states, part_idxs, sim_to_april_mat, april_to_robot, ee_pose, device, body_pose_robot
        )

        gripper_open_thr = config["robot"]["max_gripper_width"]["square_table"] - 0.005

        push_xy = push_target[:2, 3]
        at_push_xy = (ee_pos[:2] - push_xy).abs().sum() < self.pos_error_threshold * 3

        # Whether the table body has been physically pushed near the push target.
        # This prevents "done"/"go_up" from firing at episode start when the EE happens
        # to start near push_xy (gripper open, EE high) before any push has occurred.
        body_near_push_xy = (body_pose_robot[:2, 3] - push_xy).abs().sum() < 0.1

        if at_push_xy:
            if gripper_width >= gripper_open_thr:
                # Gripper open at push target: done/go_up only if table was actually pushed.
                if body_near_push_xy:
                    if ee_pos[2] >= 0.09:
                        return "done"
                    return "go_up"
                # Gripper open but table not pushed yet → episode-start false positive;
                # fall through to the grasp/pick states below.
            else:
                # Gripper not fully open at push target → open gripper to release table.
                return "release"

        grasp_target = self._get_grasp_target(body_pose_april, april_to_robot, ee_pos, device)
        grasp_xy = grasp_target[:2, 3]

        # "push": gripper fully closed — table body is grasped, move to push target.
        # Do NOT require EE to have already moved away from grasp_xy: right after pick_body
        # succeeds, EE is still at grasp_xy but the correct next action is already "push".
        if gripper_width <= self.body_grip_width + 0.005:
            return "push"

        # "pick_body": EE at body Z, gripper closing (not yet fully closed)
        body_z = body_pose_robot[2, 3]
        if abs(ee_pos[2] - body_z) < self.pos_error_threshold * 2:
            return "pick_body"

        # "reach_body_grasp_z": EE at body XY, needs to descend to body Z
        if (ee_pos[:2] - grasp_xy).abs().sum() < self.pos_error_threshold * 4:
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
            clean_target = self._get_grasp_target(body_pose_april, april_to_robot, ee_pos, device)
            target = self._add_noise(clean_target)
            result = self.satisfy(ee_pose, target, max_len=300)
            if result == "TIMEOUT":
                timeout_failure = True

        elif state == "reach_body_grasp_z":
            grasp_target = self._get_grasp_target(body_pose_april, april_to_robot, ee_pos, device)
            target_pos = grasp_target[:3, 3].clone()
            target_pos[2] = body_pose_robot[2, 3]
            target_ori = grasp_target[:3, :3]

            clean_target = C.to_homogeneous(target_pos, target_ori)
            target = self._add_noise(
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
            target = self._add_noise(clean_target, pos_std=0.003, ori_std_deg=3.0)
            self.gripper_action = 1
            result = self.gripper_less(gripper_width, self.body_grip_width)
            if result == "TIMEOUT":
                timeout_failure = True

        elif state == "push":
            clean_target = self._get_push_target(
                rb_states, part_idxs, sim_to_april_mat, april_to_robot, ee_pose, device, body_pose_robot
            )
            target = self._add_noise(clean_target, pos_std=0.001)
            result = self.satisfy(ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.5, max_len=300)
            if result == "TIMEOUT":
                timeout_failure = True

        elif state == "release":
            clean_target = self._get_push_target(
                rb_states, part_idxs, sim_to_april_mat, april_to_robot, ee_pose, device, body_pose_robot
            )
            target = self._add_noise(clean_target, pos_std=0.003, ori_std_deg=3.0)
            self.gripper_action = -1
            result = self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["square_table"] - 0.001,
            )
            if result == "TIMEOUT":
                timeout_failure = True

        elif state == "go_up":
            push_target = self._get_push_target(
                rb_states, part_idxs, sim_to_april_mat, april_to_robot, ee_pose, device, body_pose_robot
            )
            target_pos = push_target[:3, 3].clone()
            target_pos[2] = 0.1
            target_ori = push_target[:3, :3]
            clean_target = C.to_homogeneous(target_pos, target_ori)
            target = self._add_noise(clean_target)
            result = self.satisfy(ee_pose, target, max_len=150)
            if result == "TIMEOUT":
                timeout_failure = True

        elif state == "done":
            self.pre_assemble_done = True
            push_target = self._get_push_target(
                rb_states, part_idxs, sim_to_april_mat, april_to_robot, ee_pose, device, body_pose_robot
            )
            target_pos = push_target[:3, 3].clone()
            target_pos[2] = 0.1
            target_ori = push_target[:3, :3]
            clean_target = C.to_homogeneous(target_pos, target_ori)
            target = self._add_noise(clean_target, pos_std=0.003, ori_std_deg=3.0)
            self.gripper_action = -1

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
