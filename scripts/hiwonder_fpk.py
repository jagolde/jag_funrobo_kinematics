from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import FiveDOFRobotTemplate


class HiWonderRobot(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()

    
    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()
        
        th1, th2, th3, th4, th5 = curr_joint_values[0], curr_joint_values[1], curr_joint_values[2], curr_joint_values[3], curr_joint_values[4]
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5


        H0_1 = ut.dh_to_matrix([th1, l1, 0, -pi/2])
        H1_2 = ut.dh_to_matrix([-pi/2 + th2, 0, l2, pi])
        H2_3 = ut.dh_to_matrix([th3, 0, l3, pi])
        H3_4 = ut.dh_to_matrix([pi/2 + th4, 0, 0, pi/2])
        H4_5 = ut.dh_to_matrix([th5, l4+l5, 0, 0])

        
        Hlist = [H0_1, H1_2, H2_3, H3_4, H4_5]

        # Calculate EE position and rotation
        H_ee = H0_1 @ H1_2 @ H2_3 @ H3_4 @ H4_5  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist


if __name__ == "__main__":
    
    model = HiWonderRobot()
    
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()