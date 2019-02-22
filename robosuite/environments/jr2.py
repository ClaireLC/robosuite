from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments import MujocoEnv

from robosuite.models.grippers import gripper_factory
from robosuite.models.robots import JR2


class JR2Env(MujocoEnv):
    """Initializes a Baxter robot environment."""

    def __init__(
        self,
        use_indicator_object=False,
        rescale_actions=True,
        bot_motion="mmp",
        **kwargs
    ):
        """
        Args:
            bot_motion (str): "static" for static base, "mmp" for mobile base

            use_indicator_object (bool): if True, sets up an indicator object that 
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in 
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes 
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes 
                in camera. False otherwise.

            control_freq (float): how many control signals to receive 
                in every second. This sets the amount of simulation time 
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            use_camera_obs (bool): if True, every observation includes a 
                rendered image.

            camera_name (str): name of camera to be rendered. Must be 
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """
        self.use_indicator_object = use_indicator_object
        self.rescale_actions = rescale_actions
        self.bot_motion = bot_motion
        super().__init__(**kwargs)

    def _load_model(self):
        """Loads robot and optionally add grippers."""
        super()._load_model()
        self.mujoco_robot = JR2()

    def _reset_internal(self):
        """Resets the pose of the arm and grippers."""
        print("RESET")
        super()._reset_internal()
        #print(self.sim.data.qpos)
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos
        #print("data\n {}".format(self.sim.data.qpos))

    def _get_reference(self):
        """Sets up references for robots, grippers, and objects."""
        super()._get_reference()

        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints)
        #print(self.robot_joints)
        self._ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]
        #print(self._ref_joint_pos_indexes)
        if self.use_indicator_object:
            ind_qpos = self.sim.model.get_joint_qpos_addr("pos_indicator")
            self._ref_indicator_pos_low, self._ref_indicator_pos_high = ind_qpos

            ind_qvel = self.sim.model.get_joint_qvel_addr("pos_indicator")
            self._ref_indicator_vel_low, self._ref_indicator_vel_high = ind_qvel

            self.indicator_id = self.sim.model.body_name2id("pos_indicator")

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_joint_pos_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("pos")
        ]

        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("vel")
        ]
        #print(self._ref_joint_vel_actuator_indexes)
    
        self.r_grip_site_id = self.sim.model.site_name2id("r_grip_site")

    def move_indicator(self, pos):
        """Moves the position of the indicator object to @pos."""
        if self.use_indicator_object:
            self.sim.data.qpos[
                self._ref_indicator_pos_low : self._ref_indicator_pos_low + 3
            ] = pos

    # Note: Overrides super
    def _pre_action(self, action):
        #print("Action: {}".format(action))
        #print("ncon: {}".format(self.sim.data.ncon))
        
        # action is an 8-dim vector (x,theta,arm joint velocities)
        # Copy the action to a list
        new_action = action.copy().tolist()

        # Transate robot's x_vel to x and y velocities for x and y actuators
        # Get indices corresponding to x,y,theta base joints
        rootwz_ind = self.sim.model.get_joint_qpos_addr("rootwz")
        rootx_ind = self.sim.model.get_joint_qpos_addr("rootx")
        rooty_ind = self.sim.model.get_joint_qpos_addr("rooty")

        theta = self.sim.data.qpos[rootwz_ind]
        new_velx = action[rootx_ind] * np.cos(theta)
        new_vely = action[rootx_ind] * np.sin(theta)
  
        # Update x velocity in new_action
        new_action[rootx_ind] = new_velx
        # Insert h velocity into new_action
        new_action.insert(1,new_vely)

        # Set x,y velocity commands to 0, to solve the problem of sliding base
        new_action[0] = 0.0
        new_action[1] = 0.0
        
        # Optionally (and by default) rescale actions to [-1, 1]. Not desirable
        # for certain controllers. They later get normalized to the control range.
        if self.rescale_actions:
            new_action = np.clip(new_action, -1, 1)

        if self.rescale_actions:
            # rescale normalized action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            applied_action = bias + weight * new_action
        else:
            applied_action = new_action

        if (self.bot_motion == "static"):
          self.sim.data.qvel[0] = 0.0
          self.sim.data.qvel[1] = 0.0
          self.sim.data.qvel[2] = 0.0
        else:
          self.sim.data.qvel[0] = weight[0] * new_velx
          self.sim.data.qvel[1] = weight[1] * new_vely
    
        self.sim.data.ctrl[:] = applied_action

        #print("{},{},{}".format(applied_action,weight[0] * new_velx,weight[1] * new_vely))
        # gravity compensation
        self.sim.data.qfrc_applied[
            self._ref_joint_vel_indexes
        ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

        #if self.use_indicator_object:
        #    self.sim.data.qfrc_applied[
        #        self._ref_indicator_vel_low : self._ref_indicator_vel_high
        #    ] = self.sim.data.qfrc_bias[
        #        self._ref_indicator_vel_low : self._ref_indicator_vel_high
        #    ]

    def _post_action(self, action):
        """Optionally performs gripper visualization after the actions."""
        ret = super()._post_action(action)
        #self._gripper_visualization()
        return ret

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        
        Important keys:
            robot-state: contains robot-centric information.
        """
        di = super()._get_observation()
        self._check_contact()

        # proprioceptive features
        di["joint_pos"] = np.array(
            [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
        )
        #print(di["joint_pos"])
        di["joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
        )
        di["r_eef_xpos"] = self._r_eef_xpos
  
        robot_states = [
            #np.sin(di["joint_pos"]),
            #np.cos(di["joint_pos"]),
            di["joint_pos"],
            di["joint_vel"],
            di["r_eef_xpos"],
        ]

        di["robot-state"] = np.concatenate(robot_states)
        return di

    @property
    def dof(self):
        """Returns the DoF of the robot (with grippers)."""
        dof = self.mujoco_robot.dof
        return dof

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        """
        Helper method to force robot joint positions to the passed values.
        """
        self.sim.data.qpos[self._ref_joint_pos_indexes] = jpos
        self.sim.forward()

    @property
    def action_spec(self):
        low = np.ones(self.dof) * -1.
        high = np.ones(self.dof) * 1.
      
        return low, high

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("right_hand")

    @property
    def _right_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame as a numpy
        array of shape (6,)
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp("right_hand").reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes[:7]]

        Jr = self.sim.data.get_body_jacr("right_hand").reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes[:7]]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot. 
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _right_hand_quat(self):
        """
        Returns eef orientation of right hand in base from of robot.
        """
        return T.mat2quat(self._right_hand_orn)

    @property
    def _right_hand_vel(self):
        """
        Returns velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[:3]

    @property
    def _right_hand_ang_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[3:]

    @property
    def _joint_positions(self):
        """Returns a numpy array of joint positions (angles), of dimension 14."""
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """Returns a numpy array of joint (angular) velocities, of dimension 14."""
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    @property
    def _r_eef_xpos(self):
        """Returns the position of the right hand in world frame."""
        return self.sim.data.site_xpos[self.r_grip_site_id]

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Using defaults.
        """
        pass

    def _check_contact(self):
        """
        Returns True if the gripper is in contact with another object.
        """
        return False
