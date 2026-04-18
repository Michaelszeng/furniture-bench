try:
    import isaacgym
    from isaacgym import gymapi, gymtorch
except ImportError as e:
    from rich import print

    print(
        """[red][Isaac Gym Import Error]
  1. You need to install Isaac Gym, if not installed.
    - Download Isaac Gym following https://clvrai.github.io/furniture-bench/docs/getting_started/installation_guide_furniture_sim.html#download-isaac-gym
    - Then, pip install -e isaacgym/python
  2. If PyTorch was imported before furniture_bench, please import torch after furniture_bench.[/red]
"""
    )
    print()
    raise ImportError(e)


from datetime import datetime
from pathlib import Path
from typing import Union

import cv2
import gym
import numpy as np
import torch
from rich import print

import furniture_bench.controllers.control_utils as C
import furniture_bench.utils.transform as T
from furniture_bench.config import ROBOT_HEIGHT, config
from furniture_bench.controllers.diffik import diffik_factory
from furniture_bench.controllers.osc import osc_factory
from furniture_bench.envs.initialization_mode import Randomness, str_to_enum
from furniture_bench.envs.observation import (
    DEFAULT_STATE_OBS,
    DEFAULT_VISUAL_OBS,
    FULL_OBS,
)

# from furniture_bench.controllers.diffik_qp import diffik_factory
from furniture_bench.furniture import furniture_factory
from furniture_bench.furniture.parts.part import Part
from furniture_bench.robot.robot_state import ROBOT_STATE_DIMS
from furniture_bench.sim_config import sim_config
from furniture_bench.utils.pose import get_mat, rot_mat

ASSET_ROOT = str(Path(__file__).parent.parent.absolute() / "assets")

# ---------------------------------------------------------------------------
# Noise enable/disable flags for scripted policy
# (for testing — override runtime no_noise flag)
# ---------------------------------------------------------------------------
# Target noise: (1) FSM per-step goal-pose jitter via _add_noise_to_target() [main jitter source];
#               (2) randomised kick multiplier and XY speed limit in _compute_delta_pos.
_ENABLE_TARGET_NOISE: bool = False
# Action noise: i.i.d. Gaussian on delta_pos and orientation
_ENABLE_ACTION_NOISE: bool = False
# Temporally correlated (OU) action noise (non-Markovian only); independent of i.i.d. noise
_ENABLE_CORR_ACTION_NOISE: bool = False
# OU smoothing factor: τ = 1/(1-alpha) steps.  0.97 → τ≈33 steps (~3 s at 10 Hz).
_CORR_NOISE_ALPHA: float = 0.95


class FurnitureSimEnv(gym.Env):
    """FurnitureSim base class."""

    def __init__(
        self,
        furniture: str,
        num_envs: int = 1,
        resize_img: bool = True,
        obs_keys=None,
        concat_robot_state: bool = False,
        manual_label: bool = False,
        manual_done: bool = False,
        headless: bool = False,
        compute_device_id: int = 0,
        graphics_device_id: int = 0,
        init_assembled: bool = False,
        np_step_out: bool = False,
        channel_first: bool = False,
        randomness: Union[str, Randomness] = "low",
        high_random_idx: int = 0,
        save_camera_input: bool = False,
        record: bool = False,
        max_env_steps: int = 3000,
        act_rot_repr: str = "quat",
        action_type: str = "delta",  # "delta" or "pos"
        ctrl_mode: str = "osc",
        ee_laser: bool = False,
        no_noise: bool = False,
        corr_noise_alpha: float = 0.8,
        non_markovian: bool = False,
        dart_amount: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            furniture (str): Specifies the type of furniture. Options are 'lamp', 'square_table', 'desk', 'drawer', 'cabinet', 'round_table', 'stool', 'chair', 'one_leg'.
            num_envs (int): Number of parallel environments.
            resize_img (bool): If true, images are resized to 224 x 224.
            obs_keys (list): List of observations for observation space (i.e., RGB-D image from three cameras, proprioceptive states, and poses of the furniture parts.)
            concat_robot_state (bool): Whether to return concatenated `robot_state` or its dictionary form in observation.
            manual_label (bool): If true, the environment reward is manually labeled.
            manual_done (bool): If true, the environment is terminated manually.
            headless (bool): If true, simulation runs without GUI.
            compute_device_id (int): GPU device ID used for simulation.
            graphics_device_id (int): GPU device ID used for rendering.
            init_assembled (bool): If true, the environment is initialized with assembled furniture.
            np_step_out (bool): If true, env.step() returns Numpy arrays.
            channel_first (bool): If true, color images are returned in channel first format [3, H, w].
            randomness (str): Level of randomness in the environment. Options are 'low', 'med', 'high'.
            high_random_idx (int): Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
            save_camera_input (bool): If true, the initial camera inputs are saved.
            record (bool): If true, videos of the wrist and front cameras' RGB inputs are recorded.
            max_env_steps (int): Maximum number of steps per episode (default: 3000).
            act_rot_repr (str): Representation of rotation for action space. Options are 'quat', 'axis', or 'rot_6d'.
            ctrl_mode (str): 'osc' (joint torque, with operation space control) or 'diffik' (joint impedance, with differential inverse kinematics control)
        """
        super(FurnitureSimEnv, self).__init__()
        self.device = torch.device("cuda", compute_device_id)

        self.scripted_timeout = [False] * num_envs
        # Furniture for each environment (reward, reset).
        self.furnitures = [furniture_factory(furniture) for _ in range(num_envs)]

        if num_envs == 1:
            self.furniture = self.furnitures[0]
        else:
            self.furniture = furniture_factory(furniture)

        self.furniture.max_env_steps = max_env_steps
        for furn in self.furnitures:
            furn.max_env_steps = max_env_steps

        self.furniture_name = furniture
        self.num_envs = num_envs
        self.obs_keys = obs_keys or DEFAULT_VISUAL_OBS
        self.robot_state_keys = [k.split("/")[1] for k in self.obs_keys if k.startswith("robot_state")]
        self.concat_robot_state = concat_robot_state
        self.pose_dim = 7
        self.resize_img = resize_img
        self.manual_label = manual_label
        self.manual_done = manual_done
        self.headless = headless
        self.ctrl_started = False
        self._image_access_started = False
        self.init_assembled = init_assembled
        self.np_step_out = np_step_out
        self.channel_first = channel_first
        self.from_skill = 0  # TODO: Skill benchmark should be implemented in FurnitureSim.
        self.randomness = str_to_enum(randomness)
        self.high_random_idx = high_random_idx
        self.last_grasp = torch.tensor([-1.0] * num_envs, device=self.device)
        self.grasp_margin = 0.02 - 0.001  # To prevent repeating open an close actions.
        self.max_gripper_width = config["robot"]["max_gripper_width"][furniture]

        self.save_camera_input = save_camera_input
        self.img_size = sim_config["camera"]["resized_img_size" if resize_img else "color_img_size"]

        # Simulator setup.
        self.isaac_gym = gymapi.acquire_gym()
        self.sim = self.isaac_gym.create_sim(
            compute_device_id,
            graphics_device_id,
            gymapi.SimType.SIM_PHYSX,
            sim_config["sim_params"],
        )

        # our flags
        self.ctrl_mode = ctrl_mode
        self.ee_laser = ee_laser
        self.no_noise = no_noise
        self.corr_noise_alpha = corr_noise_alpha
        self.non_markovian = non_markovian
        self._corr_noise_state = {}  # (env_idx, part_idx) -> {'pos': Tensor(3), 'aa': ndarray(3)}
        # Per-env Non-Markovian pause state (used by get_assembly_action)
        self._nm_pause_remaining = [0] * num_envs  # steps left in current pause
        self._nm_prev_state_key = [None] * num_envs  # (assemble_idx, part_name, fsm_state) of last FSM call
        self._nm_pause_gripper = [-1.0] * num_envs  # gripper command to hold during pause
        self.dart_amount = dart_amount
        for furn in self.furnitures:
            for part in furn.parts:
                if no_noise:
                    part.no_noise = True
                if not _ENABLE_TARGET_NOISE:
                    part.no_iid_target_noise = True
        if self.furniture not in self.furnitures:
            for part in self.furniture.parts:
                if no_noise:
                    part.no_noise = True
                if not _ENABLE_TARGET_NOISE:
                    part.no_iid_target_noise = True
        if dart_amount != 1.0:
            for furn in self.furnitures:
                for part in furn.parts:
                    part.dart_amount = dart_amount
            if self.furniture not in self.furnitures:
                for part in self.furniture.parts:
                    part.dart_amount = dart_amount
        for furn in self.furnitures:
            for part in furn.parts:
                part.non_markovian = non_markovian
        if self.furniture not in self.furnitures:
            for part in self.furniture.parts:
                part.non_markovian = non_markovian

        self._create_ground_plane()
        self._setup_lights()
        self.import_assets()
        self.create_envs()
        self.set_viewer()
        self.set_camera()
        self.acquire_base_tensors()

        self.isaac_gym.prepare_sim(self.sim)
        self.refresh()

        self.isaac_gym.refresh_actor_root_state_tensor(self.sim)

        self.init_ee_pos, self.init_ee_quat = self.get_ee_pose()

        gym.logger.set_level(gym.logger.INFO)

        self.record = record
        if self.record:
            record_dir = Path("sim_record") / datetime.now().strftime("%Y%m%d-%H%M%S")
            record_dir.mkdir(parents=True, exist_ok=True)
            self.video_writer = cv2.VideoWriter(
                str(record_dir / "video.mp4"),
                cv2.VideoWriter_fourcc(*"MP4V"),
                30,
                (self.img_size[1] * 2, self.img_size[0]),  # Wrist and front cameras.
            )

        if act_rot_repr != "quat" and act_rot_repr != "axis" and act_rot_repr != "rot_6d":
            raise ValueError(f"Invalid rotation representation: {act_rot_repr}")
        self.act_rot_repr = act_rot_repr
        self.action_type = action_type

        # Create the action space limits on device here to save computation.
        self.act_low = torch.from_numpy(self.action_space.low).to(device=self.device)
        self.act_high = torch.from_numpy(self.action_space.high).to(device=self.device)
        self.sim_steps = int(
            1.0 / config["robot"]["hz"] / sim_config["sim_params"].dt / sim_config["sim_params"].substeps + 0.1
        )

        self.robot_state_as_dict = kwargs.get("robot_state_as_dict", True)
        self.squeeze_batch_dim = kwargs.get("squeeze_batch_dim", False)

        self.delta_pos_gain = 2.5  # Scales goal-error → delta_pos action; halved during soft states.
        self.delta_quat_gain = 1.0  # Slerp fraction for delta_quat; halved during soft states.

    def _create_ground_plane(self):
        """Creates ground plane."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.isaac_gym.add_ground(self.sim, plane_params)

    def _setup_lights(self):
        for light in sim_config["lights"]:
            l_color = gymapi.Vec3(*light["color"])
            l_ambient = gymapi.Vec3(*light["ambient"])
            l_direction = gymapi.Vec3(*light["direction"])
            self.isaac_gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)

    def create_envs(self):
        table_pos = gymapi.Vec3(0.8, 0.8, 0.4)
        self.franka_pose = gymapi.Transform()

        table_half_width = 0.015
        table_surface_z = table_pos.z + table_half_width
        self.franka_pose.p = gymapi.Vec3(0.5 * -table_pos.x + 0.1, 0, table_surface_z + ROBOT_HEIGHT)

        self.franka_from_origin_mat = get_mat(
            [self.franka_pose.p.x, self.franka_pose.p.y, self.franka_pose.p.z],
            [0, 0, 0],
        )
        self.base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]

        franka_link_dict = self.isaac_gym.get_asset_rigid_body_dict(self.franka_asset)
        self.franka_ee_index = franka_link_dict["k_ee_link"]
        self.franka_base_index = franka_link_dict["panda_link0"]
        # Parts assets.
        # Create assets.
        self.part_assets = {}
        for part in self.furniture.parts:
            asset_option = sim_config["asset"][part.name]
            self.part_assets[part.name] = self.isaac_gym.load_asset(self.sim, ASSET_ROOT, part.asset_file, asset_option)
        # Create envs.
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.env_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self.handles = {}
        self.ee_idxs = []
        self.ee_handles = []
        self.osc_ctrls = []
        self.diffik_ctrls = []

        self.base_idxs = []
        self.part_idxs = {}
        self.franka_handles = []
        for i in range(self.num_envs):
            env = self.isaac_gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            # Add workspace (table)
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.0, 0.0, table_pos.z)

            table_handle = self.isaac_gym.create_actor(env, self.table_asset, table_pose, "table", i, 0)
            table_props = self.isaac_gym.get_actor_rigid_shape_properties(env, table_handle)
            # My sus dataset processing scripts sometimes remove the table collision geometry so this check is needed
            if table_props:
                table_props[0].friction = sim_config["table"]["friction"]
            self.isaac_gym.set_actor_rigid_shape_properties(env, table_handle, table_props)

            self.base_tag_pose = gymapi.Transform()
            base_tag_pos = T.pos_from_mat(config["robot"]["tag_base_from_robot_base"])
            self.base_tag_pose.p = self.franka_pose.p + gymapi.Vec3(base_tag_pos[0], base_tag_pos[1], -ROBOT_HEIGHT)
            self.base_tag_pose.p.z = table_surface_z
            base_tag_handle = self.isaac_gym.create_actor(
                env, self.base_tag_asset, self.base_tag_pose, "base_tag", i, 0
            )
            bg_pos = gymapi.Vec3(-0.8, 0, 0.75)
            bg_pose = gymapi.Transform()
            bg_pose.p = gymapi.Vec3(bg_pos.x, bg_pos.y, bg_pos.z)
            bg_handle = self.isaac_gym.create_actor(env, self.background_asset, bg_pose, "background", i, 0)
            # TODO: Make config
            obstacle_pose = gymapi.Transform()
            obstacle_pose.p = gymapi.Vec3(self.base_tag_pose.p.x + 0.37 + 0.01, 0.0, table_surface_z + 0.015)
            obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.5 * np.pi)

            obstacle_handle = self.isaac_gym.create_actor(
                env, self.obstacle_front_asset, obstacle_pose, "obstacle_front", i, 0
            )
            part_idx = self.isaac_gym.get_actor_rigid_body_index(env, obstacle_handle, 0, gymapi.DOMAIN_SIM)
            if self.part_idxs.get("obstacle_front") is None:
                self.part_idxs["obstacle_front"] = [part_idx]
            else:
                self.part_idxs["obstacle_front"].append(part_idx)
            # Set obstacle friction.
            if sim_config.get("obstacle") is not None:
                obstacle_props = self.isaac_gym.get_actor_rigid_shape_properties(env, obstacle_handle)
                obstacle_props[0].friction = sim_config["obstacle"]["friction"]
                self.isaac_gym.set_actor_rigid_shape_properties(env, obstacle_handle, obstacle_props)

            for j, name in enumerate(["obstacle_right", "obstacle_left"]):
                y = -0.175 if j == 0 else 0.175
                obstacle_pose = gymapi.Transform()
                obstacle_pose.p = gymapi.Vec3(
                    self.base_tag_pose.p.x + 0.37 + 0.01 - 0.075,
                    y,
                    table_surface_z + 0.015,
                )
                obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.5 * np.pi)

                obstacle_handle = self.isaac_gym.create_actor(env, self.obstacle_side_asset, obstacle_pose, name, i, 0)
                part_idx = self.isaac_gym.get_actor_rigid_body_index(env, obstacle_handle, 0, gymapi.DOMAIN_SIM)
                if self.part_idxs.get(name) is None:
                    self.part_idxs[name] = [part_idx]
                else:
                    self.part_idxs[name].append(part_idx)
                # Set obstacle friction.
                if sim_config.get("obstacle") is not None:
                    obstacle_props = self.isaac_gym.get_actor_rigid_shape_properties(env, obstacle_handle)
                    obstacle_props[0].friction = sim_config["obstacle"]["friction"]
                    self.isaac_gym.set_actor_rigid_shape_properties(env, obstacle_handle, obstacle_props)

            # Add robot.
            franka_handle = self.isaac_gym.create_actor(env, self.franka_asset, self.franka_pose, "franka", i, 0)
            self.franka_num_dofs = self.isaac_gym.get_actor_dof_count(env, franka_handle)

            self.isaac_gym.enable_actor_dof_force_sensors(env, franka_handle)
            self.franka_handles.append(franka_handle)

            # Get global index of hand and base.
            self.ee_idxs.append(
                self.isaac_gym.get_actor_rigid_body_index(env, franka_handle, self.franka_ee_index, gymapi.DOMAIN_SIM)
            )
            self.ee_handles.append(self.isaac_gym.find_actor_rigid_body_handle(env, franka_handle, "k_ee_link"))
            self.base_idxs.append(
                self.isaac_gym.get_actor_rigid_body_index(env, franka_handle, self.franka_base_index, gymapi.DOMAIN_SIM)
            )
            # Set dof properties.
            franka_dof_props = self.isaac_gym.get_asset_dof_properties(self.franka_asset)
            if self.ctrl_mode == "osc":
                franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
                franka_dof_props["stiffness"][:7].fill(0.0)
                franka_dof_props["damping"][:7].fill(0.0)
                franka_dof_props["friction"][:7] = sim_config["robot"]["arm_frictions"]
            else:
                franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
                franka_dof_props["stiffness"][:7].fill(1000.0)
                franka_dof_props["damping"][:7].fill(200.0)
            # Grippers
            franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_EFFORT)
            franka_dof_props["stiffness"][7:].fill(0)
            franka_dof_props["damping"][7:].fill(0)
            franka_dof_props["friction"][7:] = sim_config["robot"]["gripper_frictions"]
            franka_dof_props["upper"][7:] = self.max_gripper_width / 2

            self.isaac_gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)
            # Set initial dof states
            franka_num_dofs = self.isaac_gym.get_asset_dof_count(self.franka_asset)
            self.default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
            self.default_dof_pos[:7] = np.array(config["robot"]["reset_joints"], dtype=np.float32)
            self.default_dof_pos[7:] = self.max_gripper_width / 2
            default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
            default_dof_state["pos"] = self.default_dof_pos
            self.isaac_gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)
            # Add furniture parts.
            poses = []
            for part in self.furniture.parts:
                pos, ori = self._get_reset_pose(part)
                part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
                part_pose = gymapi.Transform()
                part_pose.p = gymapi.Vec3(part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3])
                reset_ori = self.april_coord_to_sim_coord(ori)
                part_pose.r = gymapi.Quat(*T.mat2quat(reset_ori[:3, :3]))
                poses.append(part_pose)
                part_handle = self.isaac_gym.create_actor(env, self.part_assets[part.name], part_pose, part.name, i, 0)
                self.handles[part.name] = part_handle

                part_idx = self.isaac_gym.get_actor_rigid_body_index(env, part_handle, 0, gymapi.DOMAIN_SIM)
                # Set properties of part.
                part_props = self.isaac_gym.get_actor_rigid_shape_properties(env, part_handle)
                part_props[0].friction = sim_config["parts"]["friction"]
                self.isaac_gym.set_actor_rigid_shape_properties(env, part_handle, part_props)

                if self.part_idxs.get(part.name) is None:
                    self.part_idxs[part.name] = [part_idx]
                else:
                    self.part_idxs[part.name].append(part_idx)

            self.parts_handles = {}
            for part in self.furniture.parts:
                self.parts_handles[part.name] = self.isaac_gym.find_actor_index(env, part.name, gymapi.DOMAIN_ENV)

        # print(f'Getting the separate actor indices for the frankas and the furniture parts (not the handles)')
        self.franka_actor_idx_all = []
        self.part_actor_idx_all = []  # global list of indices, when resetting all parts
        self.part_actor_idx_by_env = {}  # allow to access part indices based on environment indices
        for env_idx in range(self.num_envs):
            self.franka_actor_idx_all.append(
                self.isaac_gym.find_actor_index(self.envs[env_idx], "franka", gymapi.DOMAIN_SIM)
            )
            self.part_actor_idx_by_env[env_idx] = []
            for part in self.furnitures[env_idx].parts:
                part_actor_idx = self.isaac_gym.find_actor_index(self.envs[env_idx], part.name, gymapi.DOMAIN_SIM)
                self.part_actor_idx_all.append(part_actor_idx)
                self.part_actor_idx_by_env[env_idx].append(part_actor_idx)

        self.franka_actor_idxs_all_t = torch.tensor(self.franka_actor_idx_all, device=self.device, dtype=torch.int32)
        self.part_actor_idxs_all_t = torch.tensor(self.part_actor_idx_all, device=self.device, dtype=torch.int32)

    def _get_reset_pose(self, part: Part):
        """Get the reset pose of the part.

        Args:
            part: The part to get the reset pose.
        """
        if self.init_assembled:
            if part.name == "chair_seat":
                # Special case handling for chair seat since the assembly of chair back is not available from initialized pose.
                part.reset_pos = [[0, 0.16, -0.035]]
                part.reset_ori = [rot_mat([np.pi, 0, 0], hom=True)]
            attached_part = False
            attach_to = None
            for assemble_pair in self.furniture.should_be_assembled:
                if part.part_idx == assemble_pair[1]:
                    attached_part = True
                    attach_to = self.furniture.parts[assemble_pair[0]]
                    break
            if attached_part:
                attach_part_pos = self.furniture.parts[attach_to.part_idx].reset_pos[0]
                attach_part_ori = self.furniture.parts[attach_to.part_idx].reset_ori[0]
                attach_part_pose = get_mat(attach_part_pos, attach_part_ori)
                if part.default_assembled_pose is not None:
                    pose = attach_part_pose @ part.default_assembled_pose
                    pos = pose[:3, 3]
                    ori = T.to_hom_ori(pose[:3, :3])
                else:
                    pos = (
                        attach_part_pose
                        @ self.furniture.assembled_rel_poses[(attach_to.part_idx, part.part_idx)][0][:4, 3]
                    )
                    pos = pos[:3]
                    ori = attach_part_pose @ self.furniture.assembled_rel_poses[(attach_to.part_idx, part.part_idx)][0]
                part.reset_pos[0] = pos
                part.reset_ori[0] = ori
            pos = part.reset_pos[self.from_skill]
            ori = part.reset_ori[self.from_skill]
        else:
            pos = part.reset_pos[self.from_skill]
            ori = part.reset_ori[self.from_skill]
        return pos, ori

    def set_viewer(self):
        """Create the viewer."""
        self.enable_viewer_sync = True
        self.viewer = None

        if not self.headless:
            self.viewer = self.isaac_gym.create_viewer(self.sim, gymapi.CameraProperties())
            # Point camera at middle env.
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.62)
            middle_env = self.envs[0]
            self.isaac_gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

    def set_camera(self):
        self.camera_handles = {}
        self.camera_obs = {}

        def create_camera(name, i):
            env = self.envs[i]
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = self.img_size[0]
            camera_cfg.height = self.img_size[1]
            camera_cfg.near_plane = 0.001
            camera_cfg.far_plane = 2.0
            camera_cfg.horizontal_fov = 40.0 if self.resize_img else 69.4
            self.camera_cfg = camera_cfg

            if name == "wrist":
                if self.resize_img:
                    camera_cfg.horizontal_fov = 55.0  # Wide view.
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                transform = gymapi.Transform()
                transform.p = gymapi.Vec3(-0.04, 0, -0.05)
                transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(-70.0))
                self.isaac_gym.attach_camera_to_body(
                    camera, env, self.ee_handles[i], transform, gymapi.FOLLOW_TRANSFORM
                )
            elif name == "front":
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                cam_pos = gymapi.Vec3(0.90, -0.00, 0.65)
                cam_target = gymapi.Vec3(-1, -0.00, 0.3)
                self.isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
                self.front_cam_pos = np.array([cam_pos.x, cam_pos.y, cam_pos.z])
                self.front_cam_target = np.array([cam_target.x, cam_target.y, cam_target.z])
            elif name == "rear":
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                transform = gymapi.Transform()
                transform.p = gymapi.Vec3(self.franka_pose.p.x + 0.08, 0, self.franka_pose.p.z + 0.2)
                transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(35.0))
                self.isaac_gym.set_camera_transform(camera, env, transform)
            return camera

        camera_names = {"1": "wrist", "2": "front", "3": "rear"}
        for env_idx, env in enumerate(self.envs):
            for k in self.obs_keys:
                if k.startswith("color"):
                    camera_name = camera_names[k[-1]]
                    render_type = gymapi.IMAGE_COLOR
                elif k.startswith("depth"):
                    camera_name = camera_names[k[-1]]
                    render_type = gymapi.IMAGE_DEPTH
                else:
                    continue
                if camera_name not in self.camera_handles:
                    self.camera_handles[camera_name] = []
                # Only when the camera handle for the current environment does not exist.
                if len(self.camera_handles[camera_name]) <= env_idx:
                    self.camera_handles[camera_name].append(create_camera(camera_name, env_idx))
                handle = self.camera_handles[camera_name][env_idx]
                tensor = gymtorch.wrap_tensor(
                    self.isaac_gym.get_camera_image_gpu_tensor(self.sim, env, handle, render_type)
                )
                if k not in self.camera_obs:
                    self.camera_obs[k] = []
                self.camera_obs[k].append(tensor)

    def import_assets(self):
        self.base_tag_asset = self._import_base_tag_asset()
        self.background_asset = self._import_background_asset()
        self.table_asset = self._import_table_asset()
        self.obstacle_front_asset = self._import_obstacle_front_asset()
        self.obstacle_side_asset = self._import_obstacle_side_asset()
        self.franka_asset = self._import_franka_asset()

    def acquire_base_tensors(self):
        # Get rigid body state tensor
        _rb_states = self.isaac_gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        _root_tensor = self.isaac_gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor)
        self.root_pos = self.root_tensor.view(self.num_envs, -1, 13)[..., 0:3]
        self.root_quat = self.root_tensor.view(self.num_envs, -1, 13)[..., 3:7]

        _forces = self.isaac_gym.acquire_dof_force_tensor(self.sim)
        _forces = gymtorch.wrap_tensor(_forces)
        self.forces = _forces.view(self.num_envs, 9)

        # Get DoF tensor
        _dof_states = self.isaac_gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)  # (num_dofs, 2), 2 for pos and vel.
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, 9)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, 9)
        # Get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.isaac_gym.acquire_jacobian_tensor(self.sim, "franka")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        # jacobian entries corresponding to franka hand
        self.jacobian_eef = self.jacobian[:, self.franka_ee_index - 1, :, :7]  # -1 due to finxed base link.
        # Prepare mass matrix tensor
        # For franka, tensor shape is (num_envs, 7 + 2, 7 + 2), 2 for grippers.
        _massmatrix = self.isaac_gym.acquire_mass_matrix_tensor(self.sim, "franka")
        self.mm = gymtorch.wrap_tensor(_massmatrix)

    def april_coord_to_sim_coord(self, april_coord_mat):
        """Converts AprilTag coordinate to simulator base_tag coordinate."""
        return self.april_to_sim_mat @ april_coord_mat

    def sim_coord_to_april_coord(self, sim_coord_mat):
        return self.sim_to_april_mat @ sim_coord_mat

    @property
    def april_to_sim_mat(self):
        return self.franka_from_origin_mat @ self.base_tag_from_robot_mat

    @property
    def sim_to_april_mat(self):
        return torch.tensor(
            np.linalg.inv(self.base_tag_from_robot_mat) @ np.linalg.inv(self.franka_from_origin_mat),
            device=self.device,
        )

    @property
    def sim_to_robot_mat(self):
        return torch.tensor(self.franka_from_origin_mat, device=self.device)

    @property
    def april_to_robot_mat(self):
        return torch.tensor(self.base_tag_from_robot_mat, device=self.device)

    @property
    def robot_to_ee_mat(self):
        return torch.tensor(rot_mat([np.pi, 0, 0], hom=True), device=self.device)

    @property
    def action_space(self):
        # Action space to be -1.0 to 1.0.
        if self.act_rot_repr == "quat":
            pose_dim = 7
        elif self.act_rot_repr == "rot_6d":
            pose_dim = 9
        else:  # axis
            pose_dim = 6

        low = np.array([-1] * pose_dim + [-1], dtype=np.float32)
        high = np.array([1] * pose_dim + [1], dtype=np.float32)

        low = np.tile(low, (self.num_envs, 1))
        high = np.tile(high, (self.num_envs, 1))

        return gym.spaces.Box(low, high, (self.num_envs, pose_dim + 1))

    @property
    def action_dimension(self):
        return self.action_space.shape[-1]

    @property
    def observation_space(self):
        low, high = -np.inf, np.inf
        parts_poses = self.furniture.num_parts * self.pose_dim
        img_size = reversed(self.img_size)
        img_shape = (3, *img_size) if self.channel_first else (*img_size, 3)

        obs_dict = {}
        robot_state = {}
        robot_state_dim = 0
        for k in self.obs_keys:
            if k.startswith("robot_state"):
                obs_key = k.split("/")[1]
                obs_shape = (ROBOT_STATE_DIMS[obs_key],)
                robot_state_dim += ROBOT_STATE_DIMS[obs_key]
                robot_state[obs_key] = gym.spaces.Box(low, high, obs_shape)
            elif k.startswith("color"):
                obs_dict[k] = gym.spaces.Box(0, 255, img_shape)
            elif k.startswith("depth"):
                obs_dict[k] = gym.spaces.Box(0, 255, img_size)
            elif k == "parts_poses":
                obs_dict[k] = gym.spaces.Box(low, high, parts_poses)
            else:
                raise ValueError(f"FurnitureSim does not support observation ({k}).")

        if robot_state:
            if self.concat_robot_state:
                obs_dict["robot_state"] = gym.spaces.Box(low, high, (robot_state_dim,))
            else:
                obs_dict["robot_state"] = gym.spaces.Dict(robot_state)

        return gym.spaces.Dict(obs_dict)

    def step_noop(self):
        """Take a no-op step."""

        # If we're doing delta control, we can simply apply a noop action:
        if self.action_type == "delta":
            noop = {
                "quat": torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32, device=self.device),
                "rot_6d": torch.tensor(
                    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                    dtype=torch.float32,
                    device=self.device,
                ),
            }[self.act_rot_repr]
            return self.step(noop)

        # Otherwise, we apply a noop action by setting temporily changing the control mode to delta control
        self.action_type = "delta"
        obs = self.step_noop()
        self.action_type = "pos"
        return obs

    @torch.no_grad()
    def step(self, action):
        """Robot takes an action.

        Args:
            action:
                (num_envs, 8): End-effector delta in [x, y, z, qx, qy, qz, qw, gripper] if self.act_rot_repr == "quat".
                (num_envs, 10): End-effector delta in [x, y, z, 6D rotation, gripper] if self.act_rot_repr == "rot_6d".
                (num_envs, 7): End-effector delta in [x, y, z, ax, ay, az, gripper] if self.act_rot_repr == "axis".
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(device=self.device)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        # Clip the action to be within the action space.
        action = torch.clamp(action, self.act_low, self.act_high)

        if not self.ctrl_started:
            self.init_ctrl()
        # Set the goal
        ee_pos, ee_quat = self.get_ee_pose()

        for env_idx in range(self.num_envs):
            if self.action_type == "delta":
                if self.act_rot_repr == "quat":
                    action_quat = action[env_idx][3:7]
                elif self.act_rot_repr == "rot_6d":
                    import pytorch3d.transforms as pt

                    # Create "actions" dataset.
                    rot_6d = action[:, 3:9]
                    rot_mat = pt.rotation_6d_to_matrix(rot_6d)

                    # pytorch3d has the real part first (w, x, y, z)
                    quat = pt.matrix_to_quaternion(rot_mat)
                    action_quat = quat[env_idx]

                    # IsaacGym expects the real part last (w, x, y, z) -> (x, y, z, w)
                    action_quat = torch.cat([action_quat[1:], action_quat[:1]])

                else:
                    action_quat = C.axisangle2quat(action[env_idx][3:6])

                if self.ctrl_mode == "osc":
                    step_ctrl = self.osc_ctrls[env_idx]
                else:
                    step_ctrl = self.diffik_ctrls[env_idx]
                step_ctrl.set_goal(
                    action[env_idx][:3] + ee_pos[env_idx],
                    C.quat_multiply(ee_quat[env_idx], action_quat).to(self.device),
                )
            elif self.action_type == "pos":
                if self.act_rot_repr == "quat":
                    action_quat = action[env_idx][3:7]
                elif self.act_rot_repr == "rot_6d":
                    import pytorch3d.transforms as pt

                    # Create "actions" dataset.
                    rot_6d = action[:, 3:9]
                    rot_mat = pt.rotation_6d_to_matrix(rot_6d)

                    # pytorch3d has the real part first (w, x, y, z)
                    quat = pt.matrix_to_quaternion(rot_mat)
                    action_quat = quat[env_idx]

                    # IsaacGym expects the real part last (w, x, y, z) -> (x, y, z, w)
                    action_quat = torch.cat([action_quat[1:], action_quat[:1]])

                else:
                    action_quat = C.axisangle2quat(action[env_idx][3:6])

                if self.ctrl_mode == "osc":
                    step_ctrl = self.osc_ctrls[env_idx]
                else:
                    step_ctrl = self.diffik_ctrls[env_idx]
                step_ctrl.set_goal(action[env_idx][:3], action_quat.to(self.device))

        for _substep in range(self.sim_steps):
            self.refresh(render_cameras=(_substep == self.sim_steps - 1))

            if self.ee_laser:
                # draw lines
                for _ in range(3):
                    noise = (np.random.random(3) - 0.5).astype(np.float32).reshape(1, 3) * 0.001
                    offset = self.franka_from_origin_mat[:-1, -1].reshape(1, 3)
                    ee_z_axis = C.quat2mat(ee_quat[env_idx]).cpu().numpy()[:, 2].reshape(1, 3)
                    line_start = ee_pos[env_idx].cpu().numpy().reshape(1, 3) + offset + noise

                    # Move the start point higher
                    line_start = line_start - ee_z_axis * 0.019

                    line_end = line_start + ee_z_axis
                    lines = np.concatenate([line_start, line_end], axis=0)
                    colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
                    self.isaac_gym.add_lines(self.viewer, self.envs[env_idx], 1, lines, colors)

            pos_action = torch.zeros_like(self.dof_pos)
            torque_action = torch.zeros_like(self.dof_pos)
            grip_action = torch.zeros((self.num_envs, 1))
            for env_idx in range(self.num_envs):
                grasp = action[env_idx, -1]
                if torch.sign(grasp) != torch.sign(self.last_grasp[env_idx]) and torch.abs(grasp) > self.grasp_margin:
                    grip_sep = self.max_gripper_width if grasp < 0 else 0.0
                    self.last_grasp[env_idx] = grasp
                else:
                    # Keep the gripper open if the grasp has not changed
                    if self.last_grasp[env_idx] < 0:
                        grip_sep = self.max_gripper_width
                    else:
                        grip_sep = 0.0

                grip_action[env_idx, -1] = grip_sep

                state_dict = {}
                ee_pos, ee_quat = self.get_ee_pose()
                state_dict["ee_pose"] = C.pose2mat(
                    ee_pos[env_idx], ee_quat[env_idx], self.device
                ).t()  # OSC expect column major
                state_dict["ee_pos"] = ee_pos[env_idx]
                state_dict["ee_quat"] = ee_quat[env_idx]
                state_dict["joint_positions"] = self.dof_pos[env_idx][:7]
                state_dict["joint_velocities"] = self.dof_vel[env_idx][:7]
                state_dict["mass_matrix"] = self.mm[env_idx][:7, :7].t()  # OSC expect column major
                state_dict["jacobian"] = self.jacobian_eef[env_idx].t()  # OSC expect column major
                state_dict["jacobian_diffik"] = self.jacobian_eef[env_idx]
                if self.ctrl_mode == "osc":
                    torque_action[env_idx, :7] = self.osc_ctrls[env_idx](state_dict)["joint_torques"]
                else:
                    pos_action[env_idx, :7] = self.diffik_ctrls[env_idx](state_dict)["joint_positions"]

                if grip_sep > 0:
                    torque_action[env_idx, 7:9] = sim_config["robot"]["gripper_torque"]
                    # pos_action[env_idx, 7:9] = sim_config["robot"]["gripper_open"]
                    pos_action[env_idx, 7:9] = self.max_gripper_width / 2
                else:
                    torque_action[env_idx, 7:9] = -sim_config["robot"]["gripper_torque"]
                    pos_action[env_idx, 7:9] = 0.0

            if self.ctrl_mode == "osc":
                self.isaac_gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torque_action))
            else:
                self.isaac_gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
                self.isaac_gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torque_action))

            # Update viewer
            self.isaac_gym.draw_viewer(self.viewer, self.sim, False)
            if not self.headless and _substep == self.sim_steps - 1:
                # self.isaac_gym.draw_viewer(self.viewer, self.sim, False)
                self.isaac_gym.sync_frame_time(self.sim)
                self.isaac_gym.clear_lines(self.viewer)

        self.isaac_gym.end_access_image_tensors(self.sim)
        self._image_access_started = False

        obs = self._get_observation()
        self.env_steps += 1

        done = self._done()
        # If scripted policy timed out for an env, force done=True for that env.
        for _env_idx in range(self.num_envs):
            if self.scripted_timeout[_env_idx]:
                done[_env_idx] = True

        return (
            obs,
            self._reward(),
            done,
            {"obs_success": True, "action_success": True},
        )

    def _reward(self):
        """Reward is 1 if two parts are assembled."""
        rewards = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

        if self.manual_label:
            # Return zeros since the reward is manually labeled by data_collector.py.
            return rewards

        # Don't have to convert to AprilTag coordinate since the reward is computed with relative poses.
        parts_poses, founds = self._get_parts_poses(sim_coord=True)
        for env_idx in range(self.num_envs):
            env_parts_poses = parts_poses[env_idx].cpu().numpy()
            env_founds = founds[env_idx].cpu().numpy()
            rewards[env_idx] = self.furnitures[env_idx].compute_assemble(env_parts_poses, env_founds)

        if self.np_step_out:
            return rewards.cpu().numpy()

        return rewards

    def _get_parts_poses(self, sim_coord=False):
        """Get furniture parts poses in the AprilTag frame.

        Args:
            sim_coord: If True, return the poses in the simulator coordinate. Otherwise, return the poses in the AprilTag coordinate.

        Returns:
            parts_poses: (num_envs, num_parts * pose_dim). The poses of all parts in the AprilTag frame.
            founds: (num_envs, num_parts). Always 1 since we don't use AprilTag for detection in simulation.
        """
        parts_poses = torch.zeros(
            (self.num_envs, len(self.furniture.parts) * self.pose_dim),
            dtype=torch.float32,
            device=self.device,
        )
        founds = torch.ones(
            (self.num_envs, len(self.furniture.parts)),
            dtype=torch.float32,
            device=self.device,
        )
        if sim_coord:
            # Return the poses in the simulator coordinate.
            for part_idx in range(len(self.furniture.parts)):
                part = self.furniture.parts[part_idx]
                rb_idx = self.part_idxs[part.name]
                part_pose = self.rb_states[rb_idx, :7]
                parts_poses[:, part_idx * self.pose_dim : (part_idx + 1) * self.pose_dim] = part_pose[
                    :, : self.pose_dim
                ]

            return parts_poses, founds

        for env_idx in range(self.num_envs):
            for part_idx in range(len(self.furniture.parts)):
                part = self.furniture.parts[part_idx]
                rb_idx = self.part_idxs[part.name][env_idx]
                part_pose = self.rb_states[rb_idx, :7]
                # To AprilTag coordinate.
                part_pose = torch.concat(
                    [
                        *C.mat2pose(
                            self.sim_coord_to_april_coord(C.pose2mat(part_pose[:3], part_pose[3:7], device=self.device))
                        )
                    ]
                )
                parts_poses[env_idx, part_idx * self.pose_dim : (part_idx + 1) * self.pose_dim] = part_pose
        return parts_poses, founds

    def get_parts_poses(self, sim_coord=False):
        return self._get_parts_poses(sim_coord=sim_coord)

    def _save_camera_input(self):
        """Saves camera images to png files for debugging."""
        root = "sim_camera"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        Path(root).mkdir(exist_ok=True)

        for cam, handles in self.camera_handles.items():
            self.isaac_gym.write_camera_image_to_file(
                self.sim,
                self.envs[0],
                handles[0],
                gymapi.IMAGE_COLOR,
                f"{root}/{timestamp}_{cam}_sim.png",
            )

            self.isaac_gym.write_camera_image_to_file(
                self.sim,
                self.envs[0],
                handles[0],
                gymapi.IMAGE_DEPTH,
                f"{root}/{timestamp}_{cam}_sim_depth.png",
            )

    def _read_robot_state(self):
        joint_positions = self.dof_pos[:, :7]
        joint_velocities = self.dof_vel[:, :7]
        joint_torques = self.forces
        ee_pos, ee_quat = self.get_ee_pose()
        for q in ee_quat:
            if q[3] < 0:
                q *= -1
        ee_pos_vel = self.rb_states[self.ee_idxs, 7:10]
        ee_ori_vel = self.rb_states[self.ee_idxs, 10:]
        gripper_width = self.gripper_width()

        robot_state_dict = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_torques": joint_torques,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "ee_pos_vel": ee_pos_vel,
            "ee_ori_vel": ee_ori_vel,
            "gripper_width": gripper_width,
        }
        return {k: robot_state_dict[k] for k in self.robot_state_keys}

    def refresh(self, render_cameras: bool = True):
        self.isaac_gym.simulate(self.sim)
        self.isaac_gym.fetch_results(self.sim, True)
        self.isaac_gym.step_graphics(self.sim)

        # Refresh tensors (needed every substep for the OSC/DiffIK controller).
        self.isaac_gym.refresh_dof_state_tensor(self.sim)
        self.isaac_gym.refresh_dof_force_tensor(self.sim)
        self.isaac_gym.refresh_rigid_body_state_tensor(self.sim)
        self.isaac_gym.refresh_jacobian_tensors(self.sim)
        self.isaac_gym.refresh_mass_matrix_tensors(self.sim)

        # Camera rendering is expensive — skip on intermediate substeps since only
        # the final frame is read as an observation.
        if render_cameras:
            self.isaac_gym.render_all_camera_sensors(self.sim)
            if self._image_access_started:
                self.isaac_gym.end_access_image_tensors(self.sim)
            self.isaac_gym.start_access_image_tensors(self.sim)
            self._image_access_started = True

    def init_ctrl(self):
        # Positional and velocity gains for robot control.
        kp = torch.tensor(sim_config["robot"]["kp"], device=self.device)
        kv = (
            torch.tensor(sim_config["robot"]["kv"], device=self.device)
            if sim_config["robot"]["kv"] is not None
            else torch.sqrt(kp) * 2.0
        )

        ee_pos, ee_quat = self.get_ee_pose()
        for env_idx in range(self.num_envs):
            self.osc_ctrls.append(
                osc_factory(
                    real_robot=False,
                    ee_pos_current=ee_pos[env_idx],
                    ee_quat_current=ee_quat[env_idx],
                    init_joints=torch.tensor(config["robot"]["reset_joints"], device=self.device),
                    kp=kp,
                    kv=kv,
                    mass_matrix_offset_val=[0.0, 0.0, 0.0],
                    position_limits=torch.tensor(config["robot"]["position_limits"], device=self.device),
                    joint_kp=10,
                )
            )

            self.diffik_ctrls.append(diffik_factory(real_robot=False))
        self.ctrl_started = True

    def get_ee_pose(self):
        """Gets end-effector pose in world coordinate."""
        hand_pos = self.rb_states[self.ee_idxs, :3]
        hand_quat = self.rb_states[self.ee_idxs, 3:7]
        base_pos = self.rb_states[self.base_idxs, :3]
        base_quat = self.rb_states[self.base_idxs, 3:7]  # Align with world coordinate.
        return hand_pos - base_pos, hand_quat

    def gripper_width(self):
        return self.dof_pos[:, 7:8] + self.dof_pos[:, 8:9]

    def _done(self) -> bool:
        dones = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        if self.manual_done:
            return dones
        for env_idx in range(self.num_envs):
            timeout = self.env_steps[env_idx] > self.furniture.max_env_steps
            if self.furnitures[env_idx].all_assembled() or timeout:
                dones[env_idx] = 1
                if timeout:
                    gym.logger.warn(f"[env] env_idx: {env_idx} timeout")
        if self.np_step_out:
            dones = dones.cpu().numpy().astype(bool)
        return dones

    def _get_color_obs(self, color_obs):
        color_obs = torch.stack(color_obs)[..., :-1]  # RGBA -> RGB
        if self.channel_first:
            color_obs = color_obs.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return color_obs

    def get_front_projection_view_matrix(self):
        cam_pos = self.front_cam_pos
        cam_target = self.front_cam_target
        width = self.img_size[0]
        height = self.img_size[1]
        near_plane = self.camera_cfg.near_plane
        far_plane = self.camera_cfg.far_plane
        horizontal_fov = self.camera_cfg.horizontal_fov

        # Compute aspect ratio
        aspect_ratio = width / height
        # Convert horizontal FOV from degrees to radians and calculate focal length
        fov_rad = np.radians(horizontal_fov)
        f = 1 / np.tan(fov_rad / 2)
        # Construct the projection matrix
        # fmt: off
        P = np.array(
            [
                [f / aspect_ratio, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far_plane + near_plane) / (near_plane - far_plane), (2 * far_plane * near_plane) / (near_plane - far_plane)],
                [0, 0, -1, 0],
            ]
        )
        # fmt: on

        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v

        forward = normalize(cam_target - cam_pos)
        up = np.array([0, 1, 0])
        right = normalize(np.cross(up, forward))
        # Recompute Up Vector
        up = np.cross(forward, right)

        # Construct the View Matrix
        # fmt: off
        V = np.matrix(
            [
                [right[0], right[1], right[2], -np.dot(right, cam_pos)],
                [up[0], up[1], up[2], -np.dot(up, cam_pos)],
                [forward[0], forward[1], forward[2], -np.dot(forward, cam_pos)],
                [0, 0, 0, 1],
            ]
        )
        # fmt: on

        return P, V

    def _get_observation(self):
        robot_state = self._read_robot_state()
        color_obs = {k: self._get_color_obs(v) for k, v in self.camera_obs.items() if "color" in k}
        depth_obs = {k: torch.stack(v) for k, v in self.camera_obs.items() if "depth" in k}

        if self.np_step_out:
            robot_state = {k: v.cpu().numpy() for k, v in robot_state.items()}
            color_obs = {k: v.cpu().numpy() for k, v in color_obs.items()}
            depth_obs = {k: v.cpu().numpy() for k, v in depth_obs.items()}

        if robot_state and self.concat_robot_state:
            if self.np_step_out:
                robot_state = np.concatenate(list(robot_state.values()), -1)
            else:
                robot_state = torch.cat(list(robot_state.values()), -1)

        if self.record:
            record_images = []
            for k in sorted(color_obs.keys()):
                img = color_obs[k][0]
                if not self.np_step_out:
                    img = img.cpu().numpy().copy()
                if self.channel_first:
                    img = img.transpose(0, 2, 3, 1)
                record_images.append(img.squeeze())
            stacked_img = np.hstack(record_images)
            self.video_writer.write(cv2.cvtColor(stacked_img, cv2.COLOR_RGB2BGR))

        obs = {}
        if isinstance(robot_state, (np.ndarray, torch.Tensor)) or robot_state:  # Check if robot_state is empty.
            if self.robot_state_as_dict:
                obs["robot_state"] = robot_state
            else:
                obs.update(robot_state)  # Flatten the dict.
        for k in self.obs_keys:
            if k == "parts_poses":
                (
                    parts_poses,
                    _,
                ) = self._get_parts_poses()  # Part poses in AprilTag coordinate.
                if self.np_step_out:
                    parts_poses = parts_poses.cpu().numpy()
                obs["parts_poses"] = parts_poses
            elif k.startswith("color"):
                obs[k] = color_obs[k]
            elif k.startswith("depth"):
                obs[k] = depth_obs[k]

        if self.squeeze_batch_dim:
            for k, v in obs.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        obs[k][kk] = vv.squeeze(0)
                else:
                    obs[k] = v.squeeze(0)
        return obs

    def get_observation(self):
        return self._get_observation()

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError
        return self._get_observation()["color_image2"]

    def is_success(self):
        return [{"task": self.furnitures[env_idx].all_assembled()} for env_idx in range(self.num_envs)]

    def reset(self):
        # can also reset the full set of robots/parts, without applying torques and refreshing
        # self._reset_franka_all()
        # self._reset_parts_all()
        for i in range(self.num_envs):
            # if using ._reset_*_all(), can set reset_franka=False and reset_parts=False in .reset_env
            self.reset_env(i)

            if self.ctrl_mode == "osc":
                # apply zero torque across the board and refresh in between each env reset (not needed if using ._reset_*_all())
                torque_action = torch.zeros_like(self.dof_pos)
                self.isaac_gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torque_action))
            self.refresh()

        self.furniture.reset()
        self.scripted_timeout = [False] * self.num_envs
        self._corr_noise_state = {}
        self._nm_pause_remaining = [0] * self.num_envs
        self._nm_prev_state_key = [None] * self.num_envs
        self._nm_pause_gripper = [-1.0] * self.num_envs

        self.refresh()

        if self.save_camera_input:
            self._save_camera_input()

        return self._get_observation()

    def reset_to(self, state):
        """Reset to a specific state.

        Args:
            state: List of observation dictionary for each environment.
        """
        for i in range(self.num_envs):
            self.reset_env_to(i, state[i])

    def reset_env(self, env_idx, reset_franka=True, reset_parts=True):
        """Resets the environment. **MUST refresh in between multiple calls
        to this function to have changes properly reflected in each environment.
        Also might want to set a zero-torque action via .set_dof_actuation_force_tensor
        to avoid additional movement**

        Args:
            env_idx: Environment index.
            reset_franka: If True, then reset the franka for this env
            reset_parts: If True, then reset the part poses for this env
        """
        self.furnitures[env_idx].reset()
        if self.randomness == Randomness.LOW and not self.init_assembled:
            self.furnitures[env_idx].randomize_init_pose(self.from_skill, pos_range=[-0.015, 0.015], rot_range=15)

        if self.randomness == Randomness.MEDIUM:
            self.furnitures[env_idx].randomize_init_pose(self.from_skill)
        elif self.randomness == Randomness.HIGH:
            self.furnitures[env_idx].randomize_high(self.high_random_idx)

        if reset_franka:
            self._reset_franka(env_idx)
        if reset_parts:
            self._reset_parts(env_idx)
        self.env_steps[env_idx] = 0
        self.scripted_timeout[env_idx] = False
        # Clear NM pause state for this env so it doesn't carry over between episodes.
        self._nm_pause_remaining[env_idx] = 0
        self._nm_prev_state_key[env_idx] = None
        self._nm_pause_gripper[env_idx] = -1.0

    def reset_env_to(self, env_idx, state):
        """Reset to a specific state. **MUST refresh in between multiple calls
        to this function to have changes properly reflected in each environment.
        Also might want to set a zero-torque action via .set_dof_actuation_force_tensor
        to avoid additional movement**

        Args:
            env_idx: Environment index.
            state: A dict containing the state of the environment.
        """
        self.furnitures[env_idx].reset()
        dof_pos = np.concatenate(
            [
                state["robot_state"]["joint_positions"],
                np.array([state["robot_state"]["gripper_width"] / 2] * 2),
            ],
        )
        self._reset_franka(env_idx, dof_pos)
        self._reset_parts(env_idx, state["parts_poses"])
        self.env_steps[env_idx] = 0
        self.scripted_timeout[env_idx] = False

    def _update_franka_dof_state_buffer(self, env_idx=None, dof_pos=None):
        """
        Sets internal tensor state buffer for Franka actor.

        Args:
            env_idx: If given, only the row for that environment is updated.
                     If None (default), all environment rows are updated.
            dof_pos: DOF positions to write. Falls back to default_dof_pos when None.
        """
        # Low randomness only.
        if self.from_skill >= 1:
            dof_pos = torch.from_numpy(self.default_dof_pos)
            ee_pos = torch.from_numpy(self.furniture.furniture_conf["ee_pos"][self.from_skill])
            ee_quat = torch.from_numpy(self.furniture.furniture_conf["ee_quat"][self.from_skill])
            dof_pos = self.robot_model.inverse_kinematics(ee_pos, ee_quat)
        else:
            dof_pos = self.default_dof_pos if dof_pos is None else dof_pos

        rows = slice(None) if env_idx is None else env_idx
        self.dof_pos[rows, 0 : self.franka_num_dofs] = torch.tensor(dof_pos, device=self.device, dtype=torch.float32)
        self.dof_vel[rows, 0 : self.franka_num_dofs] = torch.tensor(
            [0] * len(self.default_dof_pos), device=self.device, dtype=torch.float32
        )

    def _reset_franka(self, env_idx, dof_pos=None):
        """
        Resets Franka actor within a single env. If calling multiple times,
        need to refresh in between calls to properly register individual env changes,
        and set zero torques on frankas across all envs to prevent the reset arms
        from moving while others are still being reset
        """
        self._update_franka_dof_state_buffer(env_idx=env_idx, dof_pos=dof_pos)

        # Update a single actor
        actor_idx = self.franka_actor_idxs_all_t[env_idx].reshape(1, 1)
        self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(actor_idx),
            len(actor_idx),
        )

    def _reset_franka_all(self, dof_pos=None):
        """
        Resets all Franka actors across all envs
        """
        self._update_franka_dof_state_buffer(dof_pos=dof_pos)

        # Update all actors across envs at once
        self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(self.franka_actor_idxs_all_t),
            len(self.franka_actor_idxs_all_t),
        )

    def _reset_parts(self, env_idx, parts_poses=None, skip_set_state=False):
        """Resets furniture parts to the initial pose.

        Args:
            env_idx (int): The index of the environment.
            parts_poses (np.ndarray): The poses of the parts. If None, the parts will be reset to the initial pose.
        """
        for part_idx, part in enumerate(self.furnitures[env_idx].parts):
            # Use the given pose.
            if parts_poses is not None:
                part_pose = parts_poses[part_idx * 7 : (part_idx + 1) * 7]

                pos = part_pose[:3]
                ori = T.to_homogeneous([0, 0, 0], T.quat2mat(part_pose[3:]))  # Dummy zero position.
            else:
                pos, ori = self._get_reset_pose(part)

            part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
            part_pose = gymapi.Transform()
            part_pose.p = gymapi.Vec3(part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3])
            reset_ori = self.april_coord_to_sim_coord(ori)
            part_pose.r = gymapi.Quat(*T.mat2quat(reset_ori[:3, :3]))
            idxs = self.parts_handles[part.name]
            idxs = torch.tensor(idxs, device=self.device, dtype=torch.int32)

            self.root_pos[env_idx, idxs] = torch.tensor(
                [part_pose.p.x, part_pose.p.y, part_pose.p.z], device=self.device
            )
            self.root_quat[env_idx, idxs] = torch.tensor(
                [part_pose.r.x, part_pose.r.y, part_pose.r.z, part_pose.r.w],
                device=self.device,
            )

        if skip_set_state:
            # Set the value for the root state tensor, but don't call isaac gym function yet (useful when resetting all at once)
            # If skip_set_state == True, then must self.refresh() to register the isaac set_actor_root_state* function
            return

        # Reset root state for actors in a single env
        part_actor_idxs = torch.tensor(self.part_actor_idx_by_env[env_idx], device=self.device, dtype=torch.int32)
        self.isaac_gym.get_sim_actor_count(self.sim)
        self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(part_actor_idxs),
            len(part_actor_idxs),
        )

    def _reset_parts_all(self, parts_poses=None):
        """Resets ALL furniture parts to the initial pose.

        Args:
            parts_poses (np.ndarray): The poses of the parts. If None, the parts will be reset to the initial pose.
        """
        for env_idx in range(self.num_envs):
            self._reset_parts(env_idx, parts_poses=parts_poses, skip_set_state=True)

        # Reset root state for actors across all envs
        self.isaac_gym.get_sim_actor_count(self.sim)
        self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(self.part_actor_idxs_all_t),
            len(self.part_actor_idxs_all_t),
        )

    def _import_base_tag_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        base_asset_file = "furniture/urdf/base_tag.urdf"
        return self.isaac_gym.load_asset(self.sim, ASSET_ROOT, base_asset_file, asset_options)

    def _import_obstacle_front_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        obstacle_asset_file = "furniture/urdf/obstacle_front.urdf"
        return self.isaac_gym.load_asset(self.sim, ASSET_ROOT, obstacle_asset_file, asset_options)

    def _import_obstacle_side_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        obstacle_asset_file = "furniture/urdf/obstacle_side.urdf"
        return self.isaac_gym.load_asset(self.sim, ASSET_ROOT, obstacle_asset_file, asset_options)

    def _import_background_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        background_asset_file = "furniture/urdf/background.urdf"
        return self.isaac_gym.load_asset(self.sim, ASSET_ROOT, background_asset_file, asset_options)

    def _import_table_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset_file = "furniture/urdf/table.urdf"
        return self.isaac_gym.load_asset(self.sim, ASSET_ROOT, table_asset_file, asset_options)

    def _import_franka_asset(self):
        self.franka_asset_file = "franka_description_ros/franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        return self.isaac_gym.load_asset(self.sim, ASSET_ROOT, self.franka_asset_file, asset_options)

    def _detect_assemble_idx(self, env_idx: int = 0) -> int:
        """Markovian: find the first unassembled pair by checking env state."""
        for idx, (i, j) in enumerate(self.furniture.should_be_assembled):
            part1_name = self.furniture.parts[i].name
            part2_name = self.furniture.parts[j].name
            rb_idx1 = self.part_idxs[part1_name][env_idx]
            rb_idx2 = self.part_idxs[part2_name][env_idx]
            part1_pose = C.to_homogeneous(
                self.rb_states[rb_idx1][:3],
                C.quat2mat(self.rb_states[rb_idx1][3:7]),
            )
            part2_pose = C.to_homogeneous(
                self.rb_states[rb_idx2][:3],
                C.quat2mat(self.rb_states[rb_idx2][3:7]),
            )
            rel_pose = torch.linalg.inv(part1_pose) @ part2_pose
            assembled_rel_poses = self.furniture.assembled_rel_poses[(i, j)]
            if not self.furniture.assembled(rel_pose.cpu().numpy(), assembled_rel_poses):
                return idx
        return len(self.furniture.should_be_assembled)  # all assembled

    def _compute_delta_pos(
        self,
        goal_pos: torch.Tensor,
        ee_pos: torch.Tensor,
        max_delta_xy: float,
        max_delta_z: float,
        noisy: bool,
    ) -> torch.Tensor:
        """Compute a clamped delta-position action from a goal position.

        Applies a gain, a nonlinear kick for large errors, then independently
        clamps XY and Z to their respective speed limits.  When noisy=True the
        kick multiplier and the XY limit are randomised; when noisy=False they
        are deterministic.
        """
        delta = goal_pos - ee_pos
        sign = delta.sign()
        delta = delta.abs() * self.delta_pos_gain
        for i in range(3):
            if delta[i] > 0.03:
                multiplier = float(np.random.normal(1.5, 0.1)) if noisy else 1.5
                delta[i] = 0.03 + (delta[i] - 0.03) * multiplier
        delta = delta * sign
        if noisy:
            xy_limit = max_delta_xy + 0.01 * torch.rand(1, device=self.device).item()
            z_limit = max_delta_z + 0.01 * torch.rand(1, device=self.device).item()
        else:
            xy_limit = max_delta_xy
            z_limit = max_delta_z
        xy_scale = (delta[:2].abs() / xy_limit).max().clamp(min=1.0)
        z_scale = (delta[2].abs() / z_limit).clamp(min=1.0)
        delta[:2] = delta[:2] / xy_scale
        delta[2] = delta[2] / z_scale
        return delta

    def get_assembly_action(self) -> torch.Tensor:
        """
        Scripted furniture assembly logic (Markovian: all state inferred from env).

        Supports num_envs >= 1.  Each environment's part FSM is driven independently
        via self.furnitures[env_idx]; structural furniture properties (should_be_assembled,
        assembled_rel_poses, assembled()) are read from the shared self.furniture template.

        Returns:
            Tuple (noisy_actions, clean_actions, skill_completes) where:
                noisy_actions  — (num_envs, 8) tensor executed in the env
                clean_actions  — (num_envs, 8) tensor recorded in the dataset
                skill_completes — list of int, one per env (0 or 1)
        """
        if self.furniture_name not in ["one_leg", "cabinet", "lamp", "round_table"]:
            raise NotImplementedError("[one_leg, cabinet, lamp, round_table] are supported for scripted agent")

        ee_pos_all, ee_quat_all = self.get_ee_pose()  # (num_envs, 3), (num_envs, 4)
        gripper_width_all = self.gripper_width()  # (num_envs, 1)
        ee_ang_vel_all = self.rb_states[self.ee_idxs, 10:13]  # (num_envs, 3)

        all_noisy = []
        all_clean = []
        all_skill = []

        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        for env_idx in range(self.num_envs):
            ee_pos = ee_pos_all[env_idx]  # (3,)
            ee_quat = ee_quat_all[env_idx]  # (4,)
            gripper_width = gripper_width_all[env_idx]  # (1,)
            ee_ang_vel = ee_ang_vel_all[env_idx]  # (3,)

            # Build a single-env view of part_idxs so the part FSMs (which use [0])
            # correctly address this env's rigid-body state indices.
            env_part_idxs = {name: [idxs[env_idx]] for name, idxs in self.part_idxs.items()}

            assemble_idx = self._detect_assemble_idx(env_idx)
            n_pairs = len(self.furniture.should_be_assembled)

            # print(f"[green][ENV {env_idx}][/green] assemble_idx={assemble_idx}/{n_pairs}, ee_z={ee_pos[2]:.3f}")

            # Move-neutral: after all pairs assembled (or between pairs), lift EE before proceeding to next pair
            if assemble_idx >= n_pairs:
                if ee_pos[2] < 0.14:
                    gripper = torch.tensor([-1], dtype=torch.float32, device=self.device)
                    goal_pos = torch.tensor([ee_pos[0], ee_pos[1], 0.15], device=self.device)
                    delta_pos = goal_pos - ee_pos
                    delta_quat = torch.tensor([0, 0, 0, 1], device=self.device)
                    action = torch.concat([delta_pos, delta_quat, gripper])
                    all_noisy.append(action)
                    all_clean.append(action)
                    all_skill.append(0)
                else:
                    action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], dtype=torch.float32, device=self.device)
                    all_noisy.append(action)
                    all_clean.append(action)
                    all_skill.append(1)
                continue

            # Get next pair of parts to assemble
            part_idx1, part_idx2 = self.furniture.should_be_assembled[assemble_idx]
            # Use per-env furniture FSM objects so each env has independent state.
            part1 = self.furnitures[env_idx].parts[part_idx1]
            part2 = self.furnitures[env_idx].parts[part_idx2]

            # Check whether to pre-assemble part1 (use cached flag set by pre_assemble() on completion)
            part1_pre_assemble_done = not hasattr(part1, "compute_pre_assemble_state") or part1.pre_assemble_done
            part2_pre_assemble_done = not hasattr(part2, "compute_pre_assemble_state") or part2.pre_assemble_done

            # ── Non-Markovian pause: hold still without calling the FSM ──────────────
            if self.non_markovian and self._nm_pause_remaining[env_idx] > 0:
                self._nm_pause_remaining[env_idx] -= 1
                hold = torch.zeros(8, dtype=torch.float32, device=self.device)
                hold[6] = 1.0  # qw=1 (identity quaternion, same layout as timeout zero_action)
                hold[7] = self._nm_pause_gripper[env_idx]
                all_noisy.append(hold)
                all_clean.append(hold)
                all_skill.append(0)
                continue
            # ─────────────────────────────────────────────────────────────────────────

            timeout_failure = False
            clean_goal_pos = clean_goal_ori = None
            if not part1_pre_assemble_done:
                # print(f"[green][ENV {env_idx}][/green] pre-assembling part1={part1.name}")
                pre_result = part1.pre_assemble(
                    ee_pos,
                    ee_quat,
                    gripper_width,
                    self.rb_states,
                    env_part_idxs,
                    self.sim_to_april_mat,
                    self.april_to_robot_mat,
                )
                if len(pre_result) == 7:
                    goal_pos, goal_ori, clean_goal_pos, clean_goal_ori, gripper, skill_complete, timeout_failure = (
                        pre_result
                    )
                else:  # Backward compatibility for older parts that don't return clean goal
                    goal_pos, goal_ori, gripper, skill_complete, timeout_failure = pre_result
                    clean_goal_pos, clean_goal_ori = goal_pos, goal_ori
            elif not part2_pre_assemble_done:
                # print(f"[green][ENV {env_idx}][/green] pre-assembling part2={part2.name}")
                pre_result = part2.pre_assemble(
                    ee_pos,
                    ee_quat,
                    gripper_width,
                    self.rb_states,
                    env_part_idxs,
                    self.sim_to_april_mat,
                    self.april_to_robot_mat,
                )
                if len(pre_result) == 7:
                    goal_pos, goal_ori, clean_goal_pos, clean_goal_ori, gripper, skill_complete, timeout_failure = (
                        pre_result
                    )
                else:  # Backward compatibility for older parts that don't return clean goal
                    goal_pos, goal_ori, gripper, skill_complete, timeout_failure = pre_result
                    clean_goal_pos, clean_goal_ori = goal_pos, goal_ori
            else:
                # print(f"[green][ENV {env_idx}][/green] fsm_step part2={part2.name}, assemble_to={part1.name}")
                fsm_result = part2.fsm_step(
                    ee_pos,
                    ee_quat,
                    gripper_width,
                    self.rb_states,
                    env_part_idxs,
                    self.sim_to_april_mat,
                    self.april_to_robot_mat,
                    self.furnitures[env_idx].parts[part_idx1].name,
                    ee_ang_vel=ee_ang_vel,
                )
                if len(fsm_result) == 7:
                    goal_pos, goal_ori, clean_goal_pos, clean_goal_ori, gripper, skill_complete, timeout_failure = (
                        fsm_result
                    )
                else:  # Backward compatibility for older parts that don't return clean goal
                    goal_pos, goal_ori, gripper, skill_complete, timeout_failure = fsm_result
                    clean_goal_pos, clean_goal_ori = goal_pos, goal_ori

            if timeout_failure:
                print(f"[green][ENV {env_idx}][/green] scripted_timeout triggered — marking episode as failure")
                self.scripted_timeout[env_idx] = True
                zero_action = torch.zeros(8, dtype=torch.float32, device=self.device)
                zero_action[6] = 1  # qw=1 (identity quaternion)
                zero_action[7] = -1  # gripper open
                all_noisy.append(zero_action)
                all_clean.append(zero_action)
                all_skill.append(0)
                continue

            # ── Non-Markovian pause: detect state transition, arm pause for next step ─
            active_part = part1 if not part1_pre_assemble_done else part2
            if self.non_markovian and active_part._NM_PAUSES:
                new_key = (assemble_idx, active_part.name, getattr(active_part, "_last_state", None))
                if new_key != self._nm_prev_state_key[env_idx]:
                    self._nm_prev_state_key[env_idx] = new_key
                    if active_part._last_state not in active_part._NM_PAUSE_EXCLUDED_STATES:
                        self._nm_pause_remaining[env_idx] = int(np.random.randint(0, active_part._NM_MAX_PAUSE + 1))
                        self._nm_pause_gripper[env_idx] = float(gripper[0].item())
            # ─────────────────────────────────────────────────────────────────────────

            # Per-env gains (set before _compute_delta_pos which reads self.delta_pos_gain).
            self.delta_pos_gain = 2.5
            self.delta_quat_gain = 1.0
            MAX_DELTA_XY = 0.13
            MAX_DELTA_Z = 0.07
            if hasattr(part2, "_last_state"):
                if part2._last_state == "screw":
                    self.delta_pos_gain = 4.0
                    self.delta_quat_gain = 1.0
                    MAX_DELTA_XY = 0.13
                    MAX_DELTA_Z = 0.07
                elif part2._last_state == "reach_table_top_z":
                    self.delta_pos_gain = 2.5
                    self.delta_quat_gain = 1.0
                    MAX_DELTA_XY = 0.13
                    MAX_DELTA_Z = 0.015

            # Compute noisy delta position from (noisy) goal
            delta_pos = self._compute_delta_pos(
                goal_pos,
                ee_pos,
                max_delta_xy=MAX_DELTA_XY,
                max_delta_z=MAX_DELTA_Z,
                noisy=_ENABLE_TARGET_NOISE and not self.no_noise,
            )
            delta_quat = C.quat_mul(C.quat_conjugate(ee_quat), goal_ori)
            delta_quat = C.quat_slerp(identity_quat, delta_quat, self.delta_quat_gain)

            # Compute clean delta position from clean goal (deterministic — no random kick or clamp noise)
            clean_delta_pos = self._compute_delta_pos(
                clean_goal_pos, ee_pos, max_delta_xy=MAX_DELTA_XY, max_delta_z=MAX_DELTA_Z, noisy=False
            )
            clean_delta_quat = C.quat_mul(C.quat_conjugate(ee_quat), clean_goal_ori)
            clean_delta_quat = C.quat_slerp(identity_quat, clean_delta_quat, self.delta_quat_gain)

            clean_action = torch.concat([clean_delta_pos, clean_delta_quat, gripper])

            # Noise scale shared by both i.i.d. and corr noise blocks.
            _current_state_no_noise = self.furnitures[env_idx].parts[part_idx2].current_state_no_noise()
            _low_noise = self.furnitures[env_idx].parts[part_idx2].current_state_low_action_noise()
            _action_noise_scale = 0.5 if _low_noise else 1.0

            # Randomly add i.i.d. noise to the executed action (not recorded).
            # Low-action-noise states get half the standard deviation and hard clips.
            if _ENABLE_ACTION_NOISE and not self.no_noise and not _current_state_no_noise:
                pos_noise = torch.normal(torch.zeros_like(delta_pos), 0.005 * _action_noise_scale * self.dart_amount)
                if _low_noise:
                    pos_clip = 0.005 * self.dart_amount
                    pos_noise = pos_noise.clamp(-pos_clip, pos_clip)
                delta_pos = delta_pos + pos_noise
                aa_noise = [
                    np.radians(np.random.normal(0, 5 * _action_noise_scale * self.dart_amount)),
                    np.radians(np.random.normal(0, 5 * _action_noise_scale * self.dart_amount)),
                    np.radians(np.random.normal(0, 5 * _action_noise_scale * self.dart_amount)),
                ]
                if _low_noise:
                    aa_clip = np.radians(4.0)
                    aa_noise = [np.clip(v, -aa_clip, aa_clip) for v in aa_noise]
                delta_quat = C.quat_multiply(
                    delta_quat,
                    torch.tensor(T.axisangle2quat(aa_noise), device=self.device),
                ).to(self.device)

            # Temporally correlated (OU) noise — independent of i.i.d. noise above.
            # Update rule (variance-preserving): z_t = alpha·z_{t-1} + √(1-alpha²)·ε_t
            # alpha=0 → pure i.i.d.; alpha→1 → slow drift with τ = 1/(1-alpha) steps.
            if _ENABLE_CORR_ACTION_NOISE and not self.no_noise and not _current_state_no_noise and self.non_markovian:
                alpha = _CORR_NOISE_ALPHA
                scale = np.sqrt(1.0 - alpha**2)
                key = (env_idx, part_idx2)
                if key not in self._corr_noise_state:
                    self._corr_noise_state[key] = {
                        "pos": torch.zeros(3, device=self.device),
                        "aa": np.zeros(3),
                    }
                cs = self._corr_noise_state[key]
                pos_std = 0.005 * _action_noise_scale * self.dart_amount
                cs["pos"] = alpha * cs["pos"] + scale * torch.normal(torch.zeros(3, device=self.device), pos_std)
                delta_pos = delta_pos + cs["pos"]
                aa_std = np.radians(5 * _action_noise_scale * self.dart_amount)
                cs["aa"] = alpha * cs["aa"] + scale * np.random.normal(0, aa_std, size=(3,))
                delta_quat = C.quat_multiply(
                    delta_quat,
                    torch.tensor(T.axisangle2quat(cs["aa"].tolist()), device=self.device),
                ).to(self.device)

            noisy_action = torch.concat([delta_pos, delta_quat, gripper])

            # For states that opt in, record the noisy action as the clean action
            # so both target noise and action noise are reflected in the dataset.
            if (
                _ENABLE_ACTION_NOISE
                and not self.no_noise
                and self.furnitures[env_idx].parts[part_idx2].current_state_clean_action_noise()
            ):
                clean_action = noisy_action

            all_noisy.append(noisy_action)
            all_clean.append(clean_action)
            all_skill.append(skill_complete)

        return torch.stack(all_noisy), torch.stack(all_clean), all_skill

    def assembly_success(self):
        return self._done().squeeze()

    def __del__(self):
        if not self.headless:
            self.isaac_gym.destroy_viewer(self.viewer)
        self.isaac_gym.destroy_sim(self.sim)

        if self.record:
            self.video_writer.release()


class FurnitureSimFullEnv(FurnitureSimEnv):
    """FurnitureSim environment with all available observations."""

    def __init__(self, **kwargs):
        super().__init__(obs_keys=FULL_OBS, **kwargs)


class FurnitureSimStateEnv(FurnitureSimEnv):
    """FurnitureSim environment with state observations."""

    def __init__(self, **kwargs):
        obs_keys = DEFAULT_STATE_OBS
        super().__init__(obs_keys=obs_keys, concat_robot_state=True, **kwargs)
