from locomotion.envs.locomotion_gym_env import LocomotionGymEnv
from locomotion.robots import robot_config
from tqdm import tqdm

import torch
import time
import numpy as np

import pickle

SWITCHED_POSITIONS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
clip = 0.20


class LocomotionWalk(LocomotionGymEnv):
    def __init__(self, gym_config, robot_class, is_render=False, on_rack=False):
        self.pickle_actions = pickle.load(open("checkpoints/actions.p", "rb"))
        self.counter = 0
        self.action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
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
        super().__init__(gym_config, robot_class, is_render, on_rack)

        current_motor_angle = np.array(self.get_motor_angles())
        desired_motor_angle = self.default_motor_angle

        # Sets the robot to stand using blending
        for t in tqdm(range(500)):
            blend_ratio = np.minimum(t / 200.0, 1)
            action = (
                1 - blend_ratio
            ) * current_motor_angle + blend_ratio * desired_motor_angle

            self._robot.Step(action)

    def step(self, action):
        self.counter += 1
        # if self.counter == 200:
        #     quit()
        start_time = time.time()
        action = np.array(action[SWITCHED_POSITIONS])
        self.action = action

        action = np.clip(action, -clip, clip)
        actions_scaled = action * 0.25
        joint_angles = action + self.default_motor_angle

        # joint_angles = self.pickle_actions[self.counter] + self.default_motor_angle
        joint_angles[0] = np.clip(joint_angles[0], -clip, clip)
        joint_angles[3] = np.clip(joint_angles[3], -clip, clip)
        joint_angles[6] = np.clip(joint_angles[6], -clip, clip)
        joint_angles[9] = np.clip(joint_angles[9], -clip, clip)
        self._robot.Step(joint_angles[SWITCHED_POSITIONS])

        while time.time() - start_time <= 1 / 60:
            pass

        observations = self.get_observation_rack()
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

    def get_observation_rack(self):
        observations = torch.cat(
            (
                torch.tensor([0.4, 0, 0], dtype=torch.float),
                torch.tensor(
                    [0, 0, 0],
                    dtype=torch.float,
                ),
                torch.tensor(self._robot.GetProjectedGravity(), dtype=torch.float),
                torch.tensor(self._robot.GetDirection(), dtype=torch.float),
                torch.tensor(
                    self.get_motor_angles() - self.default_motor_angle[SWITCHED_POSITIONS],
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
