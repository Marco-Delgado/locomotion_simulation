from locomotion.envs.locomotion_gym_env import LocomotionGymEnv

import numpy as np
import pybullet_data as pd

SWITCHED_POSITIONS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
# STANDING_POSITION = np.array([0, 0.432, -0.77] * 4)
STANDING_POSITION = np.array([0, np.deg2rad(40), -np.deg2rad(80)] * 4)
JOINT_LIMITS_ENERGY = np.array([0.15, 0.4, 0.4] * 4)


class LocomotionWalk(LocomotionGymEnv):
    def __init__(self, gym_config, robot_class=None, is_render=False, on_rack=False):
        super().__init__(gym_config, robot_class, is_render, on_rack)

    def step(self, action):
        action = np.array(action)

        deltas = action # * JOINT_LIMITS_ENERGY + STANDING_POSITION

        self._robot.Step(deltas)

        observations = self._get_observation()
        # observations = np.concatenate(list(observations.values()))

        return observations, 0, False, {}
