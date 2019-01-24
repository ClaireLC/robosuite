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

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()


    def reward(self, action):
        """
        Reward function for the task.

          1. the agent only gets the lifting reward when flipping no more than 30 degrees.
          2. the lifting reward is smoothed and ranged from 0 to 2, capped at 2.0.
             the initial lifting reward is 0 when the pot is on the table;
             the agent gets the maximum 2.0 reward when the pot’s height is above a threshold.
          3. the reaching reward is 0.5 when the left gripper touches the left handle,
             or when the right gripper touches the right handle before the gripper geom
             touches the handle geom, and once it touches we use 0.5
        """
        reward = 0

        return reward

    @property
    def _handle_1_xpos(self):
        """Returns the position of the first handle."""
        return self.sim.data.site_xpos[self.handle_1_site_id]

    @property
    def _handle_2_xpos(self):
        """Returns the position of the second handle."""
        return self.sim.data.site_xpos[self.handle_2_site_id]

    @property
    def _pot_quat(self):
        """Returns the orientation of the pot."""
        return T.convert_quat(self.sim.data.body_xquat[self.cube_body_id], to="xyzw")

    @property
    def _world_quat(self):
        """World quaternion."""
        return T.convert_quat(np.array([1, 0, 0, 0]), to="xyzw")

    @property
    def _l_gripper_to_handle(self):
        """Returns vector from the left gripper to the handle."""
        return self._handle_1_xpos - self._l_eef_xpos

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
