from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.baxter import BaxterEnv
from robosuite.environments.jr2 import JR2Env

from robosuite.models.objects import DoorPullNoLatchObject, DoorPullWithLatchObject, DoorPullNoLatchRoomObject
from robosuite.models.arenas import TableArena, EmptyArena
from robosuite.models.robots import Baxter
from robosuite.models.tasks import DoorTask
from robosuite.models import MujocoWorldBase


class JR2Door(JR2Env):
    """
    This class corresponds to door opening task for JR2.
    """

    def __init__(
        self,
        use_object_obs=True,
        reward_shaping=True,
        door_type="dpnl",
        door_pos = [1.3,-0.05,1.0],
        door_quat = [1, 0, 0, -1],
        arena="e",
        robot_pos=[0,0,0],
        dist_to_handle_coef=1.0,
        door_angle_coef=1.0,
        handle_con_coef=1.0,
        body_door_con_coef=0.0,
        self_con_coef=0.0,
        arm_handle_con_coef=0.0,
        arm_door_con_coef=0.0,
        force_coef=0.0,
        **kwargs
    ):
        """
        Args:
            use_object_obs (bool): if True, include object (pot) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.
  
            door_type (str): type of door (pull no latch, pull with latch, push no latch, push with latch)
    
            door_pos ([x,y,z]): position of door
   
            door_quat ([w,x,y,z]): quaternion of door

            arena (str): empty or room

            robot_pos ([x,y,x]): position of robot
  
            dist_to_handle_coef: reward coefficient for eef distance to handle

            door_angle_coef: reward coefficient for angle of door
  
            handle_con_coef: reward coefficient for eef contact with handle

            body_door_con_coef: reward coefficent to penalize body contact with door

            self_con_coef: reward coefficient to penalize self collisions

        Inherits the JR2 environment; refer to other parameters described there.
        """

        # initialize the door
        if (door_type == "dpnl"):
         self.door = DoorPullNoLatchObject()
        elif (door_type == "dpwl"):
         self.door = DoorPullWithLatchObject()
        elif (door_type == "dpnlr"):
         self.door = DoorPullNoLatchRoomObject()

        self.mujoco_objects = OrderedDict([("Door", self.door)])

        self.door_pos = door_pos
        self.door_quat = door_quat
        self.robot_pos = robot_pos
        self.arena = arena

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping
        self.dist_to_handle_coef = dist_to_handle_coef
        self.door_angle_coef = door_angle_coef
        self.handle_con_coef = handle_con_coef
        self.body_door_con_coef = body_door_con_coef
        self.self_con_coef = self_con_coef
        self.arm_handle_con_coef = arm_handle_con_coef
        self.arm_door_con_coef  = arm_door_con_coef
        self.force_coef = force_coef

        super().__init__(
            **kwargs
        )

    def _load_model(self):
        """
        Loads the arena and pot object.
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos(self.robot_pos)

        # load model for table top workspace
        self.model = MujocoWorldBase()
        if (self.arena == "e"):
          self.mujoco_arena = EmptyArena()

        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()
        
        self.model = DoorTask(
          self.mujoco_arena,
          self.mujoco_robot,
          self.mujoco_objects,
        )
        
        self.model.place_objects(self.door_pos,self.door_quat)
  
    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flattened array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.door_body_id = self.sim.model.body_name2id("door")
        self.door_latch_id = self.sim.model.body_name2id("latch")
        self.door_handle_site_id = self.sim.model.site_name2id("door_handle")
        self.door_hinge_joint_id = self.sim.model.joint_name2id("door_hinge")
        #print(self.sim.model.body_names)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()


    def reward(self, action):
        """
        Reward function for the task.
        """
        reward = 0

        # Distance to door
        distance_to_handle = self._eef_distance_to_handle
        #print("distance to door: {}".format(distance_to_handle))
        #print("R: distance to door: {}".format(distance_to_handle))

        # Angle of door body (in door object frame)
        door_hinge_angle = self._door_hinge_pos

        # Penalize self contacts (arm with body)
        self_con = self.find_contacts(self.mujoco_robot.arm_contact_geoms,self.mujoco_robot.body_contact_geoms) 
        self_con_num = len(list(self_con)) > 0
        #print(self_con_num)

        # Contact with door handle
        door_handle_con = self.find_contacts(self.mujoco_robot.gripper_contact_geoms,self.mujoco_objects["Door"].handle_contact_geoms)
        door_handle_con_num = len(list(door_handle_con)) > 0
        #print("eef handle con num {}".format(door_handle_con_num))
      
        # Body to door contacts
        body_door_con = self.find_contacts(self.mujoco_robot.body_contact_geoms, self.mujoco_objects["Door"].door_contact_geoms + self.mujoco_objects["Door"].handle_contact_geoms)
        body_door_con_num = len(list(body_door_con)) > 0
        #print("body door con num {}".format(body_door_con_num))
  
        # Arm to door contacts
        arm_door_con = self.find_contacts(self.mujoco_robot.arm_contact_geoms + self.mujoco_robot.gripper_contact_geoms, self.mujoco_objects["Door"].door_contact_geoms)
        arm_door_con_num = len(list(arm_door_con)) > 0
        #print("arm door con num {}".format(arm_door_con_num))

        # Arm links to door handle contacts 
        arm_handle_con = self.find_contacts(self.mujoco_robot.arm_contact_geoms, self.mujoco_objects["Door"].handle_contact_geoms)
        arm_handle_con_num = len(list(arm_handle_con)) > 0
        #print(arm_handle_con_num)

        # Penalize large forces
        if ((abs(self._eef_force_measurement) > 60).any()):
          rew_eef_force = self.force_coef
          print("LARGE FORCE")
        else:
          rew_eef_force = 0
  
        #print("handle xpos: {}".format(self._door_handle_xpos))

        # Reward for going through door
        #base_target_pos_x = self.door_pos[0] + 2
        #base_to_target_dist = self._joint_positions[0] - base_target_pos_x
        #rew_base_to_targ = 1 - base_to_target_dist
        
        rew_dist_to_handle = self.dist_to_handle_coef * (1 - np.tanh(5*distance_to_handle))
        rew_door_angle     = self.door_angle_coef * door_hinge_angle
        rew_handle_con     = self.handle_con_coef * door_handle_con_num
        rew_body_door_con  = self.body_door_con_coef * body_door_con_num
        rew_self_con       = self.self_con_coef * self_con_num
        rew_arm_handle_con = self.arm_handle_con_coef * arm_handle_con_num
        rew_arm_door_con   = self.arm_door_con_coef * arm_door_con_num
        #print(self.arm_handle_con_coef)

        reward = rew_dist_to_handle + rew_door_angle + rew_handle_con + rew_body_door_con + rew_self_con + rew_eef_force + rew_arm_door_con

        #print("(dist_to_handle,door_angle,handle_con,body_door_con,self_con,arm_handle_con,eef_force,arm_door_con)\n({},{},{},{},{},{},{},{})".format(rew_dist_to_handle, rew_door_angle,rew_handle_con, rew_body_door_con, rew_self_con,rew_arm_handle_con,rew_eef_force,rew_arm_door_con))
        #print("total reward: {}".format(reward))

        return reward
    
    @property
    def _door_xpos(self):
        """ Returns the position of the door """
        return self.sim.data.body_xpos[self.door_body_id]
    
    @property
    def _door_handle_xpos(self):
        """ Returns position of door handle target site """
        return self.sim.data.site_xpos[self.door_handle_site_id]

    @property
    def _door_latch_xquat(self):
        """ Returns angle of door latch """
        return self.sim.data.body_xquat[self.door_latch_id]

    @property
    def _door_hinge_pos(self):
        """ Returns angle of door hinge joint """
        return self.sim.data.qpos[self.door_hinge_joint_id]

    @property
    def _eef_distance_to_handle(self):
        """ Returns vector from robot to door handle """
        dist = np.linalg.norm(self._door_handle_xpos - self._r_eef_xpos )
        return dist 

    @property
    def _world_quat(self):
        """World quaternion."""
        return T.convert_quat(np.array([1, 0, 0, 0]), to="xyzw")

    @property
    def _r_gripper_to_handle(self):
        """Returns vector from the right gripper to the handle."""
        return self._handle_2_xpos - self._r_eef_xpos

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
 
        # Object information
        if self.use_object_obs:
          # position and rotation of object in world frame
          door_pos = self.sim.data.body_xpos[self.door_body_id]
          door_quat = T.convert_quat(self.sim.data.body_xquat[self.door_body_id], to="xyzw")
          #print("door pos: {}".format(door_quat))

          di["door_pos"] = door_pos
          di["door_quat"] = door_quat
          di["door_handle_pos"] = self._door_handle_xpos 
          #di["eef_to_handle"] = self._door_handle_xpos - self._r_eef_xpos 
          di["handle_quat"] =  self._door_latch_xquat
          #print(di["handle_quat"])
    
          di["object_state"] = np.concatenate(
            [
              di["door_pos"],
              di["door_quat"],
              di["door_handle_pos"],
              #di["eef_to_handle"],
              di["handle_quat"],
            ]
          )
 
        #print("object state obs {}".format(di))
        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = super()._check_contact()
        return collision

    def _check_success(self):
        """
        Returns True if task is successfully completed
        """
        # cube is higher than the table top above a margin
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_full_size[2]
        return cube_height > table_height + 0.10
