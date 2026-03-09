from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    KinovaRobotTemplate,
)


class KinovaRobot(KinovaRobotTemplate):
    def __init__(self):
        super().__init__()

    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()

        th1, th2, th3, th4, th5, th6 = curr_joint_values
        l1, l2, l3, l4, l5, l6, l7 = (
            self.l1,
            self.l2,
            self.l3,
            self.l4,
            self.l5,
            self.l6,
            self.l7,
        )

        dh_b0 = [0, 0, 0, np.pi]

        dh_01 = [th1, -(l1 + l2), 0, (np.pi / 2)]

        dh_12 = [th2 - (np.pi / 2), 0, l3, np.pi]

        dh_23 = [th3 - (np.pi / 2), 0, 0, (np.pi / 2)]

        dh_34 = [th4, -(l4 + l5), 0, -(np.pi / 2)]

        dh_45 = [th5, 0, 0, (np.pi / 2)]

        dh_56 = [th6, -(l6 + l7), 0, np.pi]

        dh_tables = [dh_b0, dh_01, dh_12, dh_23, dh_34, dh_45, dh_56]
        H_matrix = np.eye(4)
        Hlist = []

        for table in dh_tables:
            H_matrix_new = ut.dh_to_matrix(table)
            H_matrix = H_matrix @ H_matrix_new
            Hlist.append(H_matrix_new)

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_matrix @ np.array([0, 0, 0, 1]))[:3]

        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_matrix[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist


if __name__ == "__main__":
    model = KinovaRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
