from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.baxter import BaxterEnv
from robosuite.environments.jr2 import JR2Env

from robosuite.models.objects import DoorWithHandleObject,CanObject,TestObject, CanVisualObject
from robosuite.models.arenas import TableArena, EmptyArena
from robosuite.models.robots import Baxter
from robosuite.models.tasks import DoorTask
from robosuite.models import MujocoWorldBase


class JR2Door(JR2Env):
    """
    This class corresponds to the bimanual lifting task for the Baxter robot.
    """

    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_object_obs=True,
        reward_shaping=True,
        **kwargs
    ):
        """
        Args:
            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_object_obs (bool): if True, include object (pot) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

        Inherits the Baxter environment; refer to other parameters described there.
        """

        # initialize the door
        #self.door = TestObject()
        self.door = DoorWithHandleObject()
        self.can = CanObject()
        self.mujoco_objects = OrderedDict([("Door", self.door)])
        #self.mujoco_objects = OrderedDict([("Can", self.can)])

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        super().__init__(
            **kwargs
            #gripper_left=gripper_type_left, gripper_right=gripper_type_right, **kwargs
        )

    def _load_model(self):
        """
        Loads the arena and pot object.
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.model = MujocoWorldBase()
        self.mujoco_arena = EmptyArena()
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()
        
        #self.model.merge(self.mujoco_arena)
        #self.model.merge(self.mujoco_robot)

        self.model = DoorTask(
          self.mujoco_arena,
          self.mujoco_robot,
          self.mujoco_objects,
        )
        
        # The sawyer robot has a pedestal, we want to align it with the table
        #self.mujoco_arena.set_origin([0.45 + self.table_full_size[0] / 2, 0, 0])

        self.model.place_objects()
  
        # Load door object
        #self.door_obj = self.door.get_collision(name="door", site=True)
        #self.model.merge(self.door)
        #self.model.merge(self.can)
        #self.model.worldbody.find(".//body[@name='left_hand']").append(self.door_obj)

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
        # Distance to door
        distance_to_handle = self._eef_distance_to_handle
        #print("distance to door: {}".format(distance_to_handle))
      
        rew_reach = (1 - np.tanh(distance_to_handle))
        #print("R: distance to door: {}".format(distance_to_handle))
        
        # Find contacts on gripper
        #r_contacts = list(self.find_contacts(
        #        ["latch","door","frame"], ["m1n6s200_end_effector","m1n6s200_link_6"]))

        #print(self.sim.data.ncon)
        #contact = self.sim.data.contact[0]
        #print("all contacts {}".format((contact.geom1)))
        #print("all contacts {}".format((contact.geom2)))
        #print("all contacts {}".format(self.sim.model.geom_id2name(contact.geom1)))
        #print("all contacts {}".format(self.sim.model.geom_id2name(contact.geom2)))
        #print("contacts {}".format(r_contacts))
  
        #print("handle xpos: {}".format(self._door_handle_xpos))
        
        reward = rew_reach
        #print("reward: {}".format(reward))

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
          di["eef_to_handle"] = self._door_handle_xpos - self._r_eef_xpos 
          di["handle_quat"] =  self._door_latch_xquat
          #print(di["handle_quat"])
    
          di["object_state"] = np.concatenate(
            [
              di["door_pos"],
              di["door_quat"],
              di["door_handle_pos"],
              di["eef_to_handle"],
              di["handle_quat"],
            ]
          )
 
        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        contact_geoms = (
            self.gripper_right.contact_geoms() + self.gripper_left.contact_geoms()
        )
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1) in contact_geoms
                or self.sim.model.geom_id2name(contact.geom2) in contact_geoms
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task is successfully completed
        """
        # cube is higher than the table top above a margin
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_full_size[2]
        return cube_height > table_height + 0.10
