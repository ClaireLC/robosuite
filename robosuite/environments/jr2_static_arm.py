from collections import OrderedDict
import numpy as np
import time

import robosuite.utils.transform_utils as T
from robosuite.environments import MujocoEnv

from robosuite.models.grippers import gripper_factory
from robosuite.models.robots import JR2StaticArm


class JR2StaticArmEnv(MujocoEnv):
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
        self.mujoco_robot = JR2StaticArm()

    def _reset_internal(self):
        """Resets the pose of the arm and grippers."""
        print("RESET")
        super()._reset_internal()
          
        # Reset base and arm 
        self.sim.data.qpos[self._ref_base_joint_pos_indexes] = self.mujoco_robot.init_base_qpos
        self.sim.data.qpos[self._ref_arm_joint_pos_indexes] = self.mujoco_robot.init_arm_qpos

    def _get_reference(self):
        """Sets up references for robots, grippers, and objects."""
        super()._get_reference()

        # indices for joints in qpos, qvel
        self.robot_base_joints = list(self.mujoco_robot.base_joints)
        self.robot_arm_joints = list(self.mujoco_robot.arm_joints)

        self._ref_base_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_base_joints
        ]
        self._ref_base_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_base_joints
        ]
        self._ref_arm_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_arm_joints
        ]
        self._ref_arm_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_arm_joints
        ]
        self._rootwz_ind = self.sim.model.get_joint_qvel_addr("rootwz")
        self._rootx_ind = self.sim.model.get_joint_qvel_addr("rootx")
        self._rooty_ind = self.sim.model.get_joint_qvel_addr("rooty")

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
    
        self.r_grip_site_id = self.sim.model.site_name2id("r_grip_site")

        self._ref_sensor_indexes = [
            self.sim.model.sensor_name2id(sensor)
            for sensor in self.sim.model.sensor_names
        ]
        self.prev_base_pos = self.sim.data.qpos[self._ref_base_joint_pos_indexes]

    def move_indicator(self, pos):
        """Moves the position of the indicator object to @pos."""
        if self.use_indicator_object:
            self.sim.data.qpos[
                self._ref_indicator_pos_low : self._ref_indicator_pos_low + 3
            ] = pos

    # Note: Overrides super
    def _pre_action(self, action):
        velx_w = self.sim.data.qvel[self._rootx_ind]
        vely_w = self.sim.data.qvel[self._rooty_ind]
        velx_robot = velx_w * np.cos(self.theta_w) + vely_w * np.sin(self.theta_w)
        vely_robot = - velx_w * np.sin(self.theta_w) + vely_w * np.cos(self.theta_w)

        #print("ncon: {}".format(self.sim.data.ncon))
        
        # action is an 8-dim vector (x,theta,arm joint velocities)
        # Copy the action to a list
        new_action = action.copy().tolist()

        print("policy action {}".format(new_action))
        # Transate robot's x_vel to x and y velocities for x and y actuators
        new_velx = action[self._rootx_ind] * np.cos(self.theta_w)
        new_vely = action[self._rootx_ind] * np.sin(self.theta_w)
  
        # Update x velocity in new_action
        new_action[self._rootx_ind] = new_velx
        # Insert h velocity into new_action
        new_action.insert(self._rooty_ind,new_vely)

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

        print("theta: {}".format(self.theta_w))
        print("robot x vel:{}".format(velx_robot))
        print("robot y vel:{}".format(vely_robot))
        print("world x vel:{}".format(velx_w))
        print("world y vel:{}".format(vely_w))
        print("world wz vel:{}".format(self.sim.data.qvel[self._rootwz_ind]))

        if (self.bot_motion == "static"):
          self.sim.data.qvel[0] = 0.0
          self.sim.data.qvel[1] = 0.0
          self.sim.data.qvel[2] = 0.0
        else:
          action_scale = 0.05
          new_velx = weight[self._rootx_ind] * applied_action[self._rootx_ind] * action_scale
          new_vely = weight[self._rooty_ind] * applied_action[self._rooty_ind] * action_scale
          new_veltheta = weight[self._rootwz_ind] * applied_action[self._rootwz_ind] * action_scale
          self.sim.data.qvel[self._rootx_ind] = new_velx
          self.sim.data.qvel[self._rooty_ind] = new_vely
          self.sim.data.qvel[self._rootwz_ind] = new_veltheta
          #print("robot y vel:{}".format(vely_robot))
          if (abs(vely_robot) > 0.0005):
            print("SLIPPING robot y vel:{}".format(vely_robot))
            self.sim.data.qpos[self._rooty_ind] = self.prev_base_pos[self._rooty_ind]
          else:
            self.prev_base_pos = self.sim.data.qpos[self._ref_base_joint_pos_indexes]

        # Set x,y velocity commands to 0, to solve the problem of sliding base
        applied_action[self._rootx_ind] = 0.0
        applied_action[self._rooty_ind] = 0.0
        applied_action[self._rootwz_ind] = 0.0

        #self.sim.data.qpos[self._ref_arm_joint_pos_indexes] = self.mujoco_robot.init_arm_qpos
        self.sim.data.qvel[self._ref_arm_joint_vel_indexes] = 0.0
        print("applied action {}".format(applied_action))
        #print("set qvels {},{},{}".format(new_velx,new_vely,new_veltheta))
        print("actual qvels {}".format(self.sim.data.qvel[self._ref_base_joint_vel_indexes]))
        self.sim.data.ctrl[:] = applied_action

        # gravity compensation
        self.sim.data.qfrc_applied[
            self._ref_arm_joint_vel_indexes
        ] = self.sim.data.qfrc_bias[self._ref_arm_joint_vel_indexes]
        self.sim.data.qfrc_applied[
            self._ref_base_joint_vel_indexes
        ] = self.sim.data.qfrc_bias[self._ref_base_joint_vel_indexes]

        #print("vely robot: {}".format(self.sim.data.qvel[rooty_ind]/np.sin(theta)))
        #print("velx_command/velx_actual")
        #print("{}/{}".format(new_velx,self.sim.data.qvel[rootx_ind]))
        #print("vely_command/vely_actual")
        #print("{}/{}".format(new_vely,self.sim.data.qvel[rooty_ind]))
        #print("veltheta_command/veltheta_actual")
        #print("{}/{}".format(applied_action[2],self.sim.data.qvel[rootwz_ind]))
        #print("arm joint positions:{}".format(self.sim.data.qpos[self._ref_arm_joint_pos_indexes]))

    def _post_action(self, action):
        """Optionally performs gripper visualization after the actions."""
        ret = super()._post_action(action)
        #velx_w = self.sim.data.qvel[self._rootx_ind]
        #vely_w = self.sim.data.qvel[self._rooty_ind]
        #velx_robot = velx_w * np.cos(self.theta_w) - vely_w * np.sin(self.theta_w)
        #vely_robot = velx_w * np.sin(self.theta_w) - vely_w * np.cos(self.theta_w)
        #print(vely_robot)
        #if (abs(vely_robot) > 0.0005):
        #  print("SLIPPING")
        #  self.sim.data.qvel[self._rootx_ind] = 0.0
        #  self.sim.data.qvel[self._rooty_ind] = 0.0
        #  self.sim.forward()
        #  vely_robot = velx_w * np.sin(self.theta_w) - vely_w * np.cos(self.theta_w)
        #  print(vely_robot)
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
            [self.sim.data.qpos[x] for x in self._ref_base_joint_pos_indexes]
        )
        #print(di["joint_pos"])
        di["joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_base_joint_vel_indexes]
        )
        di["r_eef_xpos"] = self._r_eef_xpos
        di["robot_pose"] = self.robot_pose_in_world.flatten()
  
        robot_states = [
            #np.sin(di["joint_pos"]),
            #np.cos(di["joint_pos"]),
            di["joint_pos"],
            di["joint_vel"],
            di["r_eef_xpos"],
            di["robot_pose"],
        ]

        di["robot-state"] = np.concatenate(robot_states)
  
        print("EEF force/torque {}/{}\n".format(self._eef_force_measurement,self._eef_torque_measurement))
        #print("robot state obs: {}".format(di))
        return di

    @property
    def dof(self):
        """Returns the DoF of the robot (with grippers)."""
        dof = self.mujoco_robot.dof
        return dof

    @property
    def theta_w(self):
        """Returns theta of robot in world frame"""
        theta = self.sim.data.qpos[self._rootwz_ind]
        return theta

    @property
    def robot_pose_in_world(self):
        pos_in_world = self.sim.data.get_body_xpos("base_footprint")
        rot_in_world = self.sim.data.get_body_xmat("base_footprint").reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)
        return pose_in_world
    
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

    @property
    def _eef_force_measurement(self):
        """Returns sensor measurement."""
        sensor_id = self.sim.model.sensor_name2id("force_ee")
        return self.sim.data.sensordata[sensor_id*3 : sensor_id*3 + 3]

    @property
    def _eef_torque_measurement(self):
        """Returns sensor measurement."""
        sensor_id = self.sim.model.sensor_name2id("torque_ee")
        return self.sim.data.sensordata[sensor_id*3 : sensor_id*3 + 3]

    @property
    def _l2_force_measurement(self):
        """Returns sensor measurement."""
        sensor_id = self.sim.model.sensor_name2id("force_2")
        return self.sim.data.sensordata[sensor_id*3 : sensor_id*3 + 3]

    @property
    def _l3_force_measurement(self):
        """Returns sensor measurement."""
        sensor_id = self.sim.model.sensor_name2id("force_3")
        return self.sim.data.sensordata[sensor_id*3 : sensor_id*3 + 3]

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
