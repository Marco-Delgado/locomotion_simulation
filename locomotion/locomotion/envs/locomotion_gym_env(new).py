"""This file implements the locomotion gym env."""
import collections
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet  # pytype: disable=import-error
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd

from locomotion.robots import robot_config
from locomotion.robots import laikago_pose_utils
from locomotion.robots import a1_robot
from locomotion.robots import a1
from locomotion.envs.sensors import sensor
from locomotion.envs.sensors import space_utils

_ACTION_EPS = 0.01
_NUM_SIMULATION_ITERATION_STEPS = 300
_LOG_BUFFER_LENGTH = 5000


class LocomotionGymEnv(gym.Env):

    def __init__(self):
        self.switched_positions = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

        self.INIT_MOTOR_ANGLES = np.array([
                                         laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
                                         laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
                                         laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE
                                     ] * 4)

        p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=30)
        p.setTimeStep(0.001)
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(enableConeFriction=0)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.loadURDF("plane.urdf")
        self._robot = a1.A1(p)

    def reset(self):
        # Reset the pose of the robot.
        self._robot.Reset(reload_urdf=False,
                          default_motor_angles=self.INIT_MOTOR_ANGLES,
                          reset_time=0.0)
        for s in self.all_sensors():
            s.on_reset(self)

        return self._get_observation()

    def step(self, action):
        # relative position (move it 5 more degrees than the previous location)

        self._last_base_position = self._robot.GetBasePosition()
        self._last_action = action

        # switching the indexing from FR, FL, RR, RL --> FL, FR, RL, RR
        action = np.array(action)
        action[idx]

        self._robot.ApplyAction(action)

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # make a dictionary
        observations = \
            {
             "joint_pos": self._robot.GetMotorAngles()[self.switched_positions],
             "joint_vel": self._robot.GetMotorVelocities()[self.switched_positions],
             "euler_rot": self._robot.GetBaseRollPitchYaw(),
             "feet_contact": self._robot.GetFootContacts()
            }

        return observations
