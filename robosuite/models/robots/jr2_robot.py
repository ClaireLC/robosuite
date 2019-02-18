import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class JR2(Robot):
    """JR2."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/jr2/jr2_with_arm.xml"))

        self.bottom_offset = np.array([0, 0, 0])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base_footprint']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 8

    @property
    def joints(self):
        return [
                "rootx",
                "rooty",
                "rootwz",
                "m1n6s200_joint_1",
                "m1n6s200_joint_2",
                "m1n6s200_joint_3",
                "m1n6s200_joint_4",
                "m1n6s200_joint_5",
                "m1n6s200_joint_6",
                #"m1n6s200_joint_finger_1",
                #"m1n6s200_joint_finger_2",
               ]

    @property
    def init_qpos(self):
        pos = np.zeros(9)
        pos[4] = np.pi - 0.1
        pos[5] = np.pi - 0.1
        return pos
    
    @property
    def visualization_sites(self):
        reutrn ["r_grip_site",]

    @property
    def body_contact_geoms(self):
        return[
          "body",
          "neck",
          "head",
          "front_caster",
          "rear_caster",
          "l_wheel_link",
          "r_wheel_link",
        ]
  
    @property
    def arm_contact_geoms(self):
        return[
          "armlink_base",
          "armlink_2",  
          "armlink_3",  
          "armlink_5",  
          "armlink_6",  
          "fingerlink_2",
          "fingertip_2",
          "fingertip_2_hook",
        ]

    @property
    def gripper_contact_geoms(self):
        return[
          "fingerlink_2",
          "fingertip_2",
          "fingertip_2_hook",
        ]
