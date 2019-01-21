import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class JR2(Robot):
    """JR2."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/JR2/jr2_with_arm.xml"))

        self.bottom_offset = np.array([0, 0, 0])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base_link']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 12

    @property
    def joints(self):
        #print(self.worldbody.find("./body/joint").get("name"))
        #return [self.worldbody.find("./body/joint").get("name")]
        return [
                "left_wheel",
                "right_wheel",
                "pan_joint",
                "tilt_joint",
                "m1n6s200_joint_1",
                "m1n6s200_joint_2",
                "m1n6s200_joint_3",
                "m1n6s200_joint_4",
                "m1n6s200_joint_5",
                "m1n6s200_joint_6",
                "m1n6s200_joint_finger_1",
                "m1n6s200_joint_finger_2",
               ]

    @property
    def init_qpos(self):
        #return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])
        return np.zeros(12)
