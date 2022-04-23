from locomotion.robots.aliengo_robot import AliengoRobot
from locomotion.envs.ros_node import RobotRosSubscriber
import numpy as np


SIM2REAL_MAPPING = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
REAL2SIM_MAPPING = SIM2REAL_MAPPING


def wrap_heading(heading):
    """Ensures input heading is between -180 an 180; can be float or np.ndarray"""
    return (heading + np.pi) % (2 * np.pi) - np.pi


class AlienGoRobotIsaac(AliengoRobot):
    def __init__(self, pybullet_client, time_step=0.002, **kwargs):
        super().__init__(pybullet_client, time_step=time_step, **kwargs)
        self.ros_sub = RobotRosSubscriber("aliengo_ros_node")

    def Step(self, sim_joint_angles, *args, **kwargs):
        # Convert FROM Isaac TO reality
        reality_joint_angles = wrap_heading(sim_joint_angles[SIM2REAL_MAPPING])
        super().Step(reality_joint_angles)

    def GetMotorAngles(self):
        reality_joint_angles = super().GetMotorAngles()
        sim_joint_angles = reality_joint_angles[SIM2REAL_MAPPING]
        return sim_joint_angles

    def GetMotorVelocities(self):
        reality_joint_velocities = super().GetMotorVelocities()
        sim_joint_velocities = reality_joint_velocities[SIM2REAL_MAPPING]
        return sim_joint_velocities

    def GetDepthObs(self):
        return self.robot_sub.front_depth_img

    def GetBaseVelocity(self):
        print("getting tracking camera lin vel")
        return self.robot_sub.base_linear_velocity

    def GetTrueBaseRollPitchYawRate(self):
        print("getting tracking camera ang vel")
        return self.robot_sub.base_angular_velocity
