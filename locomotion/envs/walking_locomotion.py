from locomotion.envs.locomotion_gym_env import LocomotionGymEnv

import torch
import time
import numpy as np

SWITCHED_POSITIONS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
clip = 100


class LocomotionWalk(LocomotionGymEnv):
    def __init__(self, gym_config, robot_class, is_render=False, on_rack=False):
        self.default_motor_angle = np.array(
            [
                0.1000,
                0.8000,
                -1.5000,
                -0.1000,
                0.8000,
                -1.5000,
                0.1000,
                1.0000,
                -1.5000,
                -0.1000,
                1.0000,
                -1.5000,
            ]
        )
        self.action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        super().__init__(gym_config, robot_class, is_render, on_rack)

    def step(self, action):
        start_time = time.time()
        self.action = action

        action = np.array(action[SWITCHED_POSITIONS])
        action = np.clip(action, -clip, clip)

        joint_angles = action + self.default_motor_angle
        self._robot.Step(joint_angles)

        while time.time() - start_time <= 1 / 60:
            pass

        observations = self.get_observation()
        observations = np.clip(observations, -clip, clip)

        return observations, 0, False, {}

    def get_observation(self):
        observations = torch.cat(
            (
                torch.tensor(self._robot.GetBaseVelocity(), dtype=torch.float),
                torch.tensor(
                    self._robot.GetTrueBaseRollPitchYawRate(),
                    dtype=torch.float,
                ),
                torch.tensor(self._robot.GetProjectedGravity(), dtype=torch.float),
                torch.tensor(self._robot.GetDirection(), dtype=torch.float),
                torch.tensor(
                    self.get_motor_angles() - self.default_motor_angle,
                    dtype=torch.float,
                ),
                torch.tensor(
                    self.get_motor_velocities(),
                    dtype=torch.float,
                ),
                torch.tensor(self.action, dtype=torch.float),
            ),
            dim=-1,
        )
        return observations

    def get_motor_angles(self):
        return self._robot.GetMotorAngles()[SWITCHED_POSITIONS]

    def get_motor_velocities(self):
        return self._robot.GetMotorVelocities()[SWITCHED_POSITIONS]
