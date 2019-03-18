import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class JR2Gripper(Robot):
    """JR2 with a two-fingered gripper."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/jr2/jr2_with_gripper.xml"))

        self.bottom_offset = np.array([0, 0, -0.01])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base_footprint']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        base_dof = 2
        arm_dof = 6
        gripper_dof = 1
        return base_dof + arm_dof + gripper_dof

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
                "m1n6s200_joint_finger_1",
                "m1n6s200_joint_finger_2",
               ]

    #@property
    def init_qpos(self,distance):
        # straight arm
        #pos = np.array([ 3.71955388e-01, -4.32114760e-02, -5.92153450e-02, -1.71517591e+00,2.83001900e+00,3.37765872e+00,1.71800951e+00,1.87382209e-02,-1.78553740e-03])

        if distance == "close":
          # bent arm far from door
          pos = np.array([ 3.70740471e-01,-0.5,-5.92161524e-02,-1.71636826e+00,2.48744571e+00,4.35466325e+00,1.68285118e+00,4.26563177e-02,-1.51617785e-03,0,0])
        elif distance == "touching_angled":
          # 45 degree body
          pos = np.array([0.41899952,-0.3610791,0.85207989,-1.99673054,1.22329899,1.67109673,5.49132849,-0.66292144,-2.96626974,0,0])
        else:
          # 45 degree body
          pos = np.array([0.41899952,-0.3610791,0.85207989,-1.99673054,1.22329899,1.67109673,5.49132849,-0.66292144,-2.96626974,0,0])
  
        return pos
    
    @property
    def visualization_sites(self):
        return ["r_grip_site",]

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
        ]

    @property
    def gripper_contact_geoms(self):
        return[
          "fingerlink_2",
          "fingerlink_1",
          #"fingerlink_2_hook",
          #"fingerlink_1_hook",
          "armlink_6_1",
        ]
