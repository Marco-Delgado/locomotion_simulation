from locomotion.envs.locomotion_gym_env import LocomotionGymEnv
from locomotion.robots import robot_config
from tqdm import tqdm

import torch
import time
import numpy as np

clip = 100
lin_vel = 2.0
ang_vel = 0.25
dof_vel = 0.05
hertz = 1 / 60


class LocomotionWalk(LocomotionGymEnv):
    def __init__(self, gym_config, robot_class, is_render=False, on_rack=False):
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
        self.action = np.zeros(12)
        super().__init__(gym_config, robot_class, is_render, on_rack)

        current_motor_angle = np.array(self.get_motor_angles())
        desired_motor_angle = self.reference_pose
        for t in tqdm(range(500)):
            blend_ratio = np.minimum(t / 200.0, 1)
            action = (
                1 - blend_ratio
            ) * current_motor_angle + blend_ratio * desired_motor_angle

            self._robot.Step(action)

    def step(self, action):
        start_time = time.time()

        self.action = np.clip(action, -clip, clip)
        joint_angles = (np.array(self.action) * 0.50 * 0.30) + self.reference_pose
        self._robot.Step(joint_angles)

        observations = self.get_observation()
        observations = torch.clip(observations, -clip, clip)

        while time.time() - start_time <= hertz:
            pass
        return observations, 0, False, {}

    def get_observation(self):
        observations = torch.cat(
            (
                torch.tensor(
                    self._robot.GetBaseVelocity() * lin_vel,
                    dtype=torch.float
                ),
                torch.tensor(
                    self._robot.GetTrueBaseRollPitchYawRate() * ang_vel,
                    dtype=torch.float,
                ),
                torch.tensor(
                    self._robot.GetProjectedGravity(),
                    dtype=torch.float,
                ),
                torch.tensor(
                    self._robot.GetDirection() * [lin_vel, lin_vel, ang_vel],
                    dtype=torch.float,
                ),
                torch.tensor(
                    self.get_motor_angles() - self.reference_pose,
                    dtype=torch.float,
                ),
                torch.tensor(
                    self.get_motor_velocities() * dof_vel,
                    dtype=torch.float,
                ),
                torch.tensor(
                    self.action,
                    dtype=torch.float),
            ),
            dim=-1,
        )
        return observations

    def get_motor_angles(self):
        return self._robot.GetMotorAngles()

    def get_motor_velocities(self):
        return self._robot.GetMotorVelocities()
