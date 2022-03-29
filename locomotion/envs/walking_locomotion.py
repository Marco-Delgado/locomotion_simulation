from locomotion.envs.locomotion_gym_env import LocomotionGymEnv

import time
import numpy as np

SWITCHED_POSITIONS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
clip = 100


class LocomotionWalk(LocomotionGymEnv):
    def __init__(self, gym_config, robot_class, is_render=False, on_rack=False):
        super().__init__(gym_config, robot_class, is_render, on_rack)

    def step(self, action):
        start_time = time.time()

        action = np.array(action[SWITCHED_POSITIONS])
        action = np.clip(action, -clip, clip)

        deltas = action #+ default_dof_pos
        self._robot.Step(deltas)

        observations = self._get_observation()
        observations = np.clip(observations, -clip, clip)

        while time.time() - start_time <= 1 / 50:
            pass

        return observations, 0, False, {}
