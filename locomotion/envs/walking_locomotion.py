from locomotion.envs.locomotion_gym_env import LocomotionGymEnv
from locomotion.robots import robot_config
from tqdm import tqdm

import torch
import time
import numpy as np

import pickle

SWITCHED_POSITIONS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
clip = 100


class LocomotionWalk(LocomotionGymEnv):
    def __init__(self, gym_config, robot_class, is_render=False, on_rack=False):
        self.pickle_obs = pickle.load(open("checkpoints/obs.p", "rb"))
        self.pickle_actions = pickle.load(open("checkpoints/actions.p", "rb"))
        self.real = np.zeros((4, 500))
        self.pickle = np.zeros((4, 500))
        self.action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.reference_pose = np.array(
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

        for stand_time in tqdm(range(100)):
            self._robot.Step(self.reference_pose)

    def step(self, action):
        start_time = time.time()

        self.action = np.clip(action, -clip, clip)

        joint_angles = np.array(self.action[SWITCHED_POSITIONS]) + self.reference_pose
        # joint_angles = (
        #     self.pickle_actions[self.count][SWITCHED_POSITIONS] + self.reference_pose
        # )

        self._robot.Step(joint_angles)

        observations = self.get_observation()
        observations = torch.clip(observations, -clip, clip)

        while time.time() - start_time <= 1 / 60:
            pass
        return observations, 0, False, {}

    def get_observation(self):
        observations = torch.cat(
            (
                torch.tensor(self._robot.GetBaseVelocity() * 2, dtype=torch.float),
                torch.tensor(
                    self._robot.GetTrueBaseRollPitchYawRate() * 0.25,
                    dtype=torch.float,
                ),
                torch.tensor(
                    self._robot.GetProjectedGravity(),
                    dtype=torch.float,
                ),
                torch.tensor(
                    self._robot.GetDirection() * [2, 2, 0.25],
                    dtype=torch.float,
                ),
                torch.tensor(
                    self.get_motor_angles() - self.reference_pose,
                    dtype=torch.float,
                ),
                torch.tensor(
                    self.get_motor_velocities() * 0.05,
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
                torch.tensor([1.7, 0, 0], dtype=torch.float),
                torch.tensor(
                    self._robot.GetTrueBaseRollPitchYawRate() * 0,
                    dtype=torch.float,
                ),
                torch.tensor(self._robot.GetProjectedGravity(), dtype=torch.float),
                torch.tensor(
                    self._robot.GetDirection() * [2, 2, 0.25], dtype=torch.float
                ),
                torch.tensor(
                    self.get_motor_angles() - self.reference_pose,
                    dtype=torch.float,
                ),
                torch.tensor(
                    self.get_motor_velocities() * 0.05,
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
