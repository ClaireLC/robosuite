import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class PR2(Robot):
    """PR2."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/PR2/pr2.xml"))

        self.bottom_offset = np.array([0, 0, -0.913])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        return ["right_j{}".format(x) for x in range(7)]

    @property
    def init_qpos(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])
