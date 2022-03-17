from locomotion.envs.locomotion_gym_env import LocomotionGymEnv

import time
import numpy as np

SWITCHED_POSITIONS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
JOINT_LIMITS_ENERGY = np.array([0.15, 0.4, 0.4] * 4)


class LocomotionWalk(LocomotionGymEnv):
    def __init__(self, gym_config, robot_class=None, is_render=False, on_rack=False):
        super().__init__(gym_config, robot_class, is_render, on_rack)
        self.STANDING_POSITION = LocomotionGymEnv.deafult_dof_pos

    def step(self, action):
        start_time = time.time()
        action = np.array(action)

        deltas = action * JOINT_LIMITS_ENERGY + self.STANDING_POSITION

        self._robot.Step(deltas)

        observations = self._get_observation()
        # observations = np.concatenate(list(observations.values()))

        while time.time() - start_time <= 1 / 50:
            pass

        return observations, 0, False, {}
