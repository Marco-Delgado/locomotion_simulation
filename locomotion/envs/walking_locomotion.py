from locomotion.envs.locomotion_gym_env import LocomotionGymEnv
from locomotion.robots import robot_config
from tqdm import tqdm

import torch
import time
import numpy as np

import pickle
import matplotlib.pyplot as plt

clip = 100


class LocomotionWalk(LocomotionGymEnv):
    def __init__(self, gym_config, robot_class, is_render=False, on_rack=False):
        self.pickle_obs = pickle.load(open("checkpoints/obs.p", "rb"))
        self.pickle_actions = pickle.load(open("checkpoints/actions.p", "rb"))
        self.torque = pickle.load(open("checkpoints/torque.p", "rb"))
        self.pickle_a = []
        self.pickle = []
        self.action = np.zeros(12)
        self.count = 0
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
        self.count += 1

        # if self.count == 248:
        #     plt.figure()
        #     plt.plot(np.arange(500), self.real[0], label="sim obs")
        #     plt.plot(np.arange(500), self.pickle[0], label="sim actions")
        #     plt.legend()
        #     plt.savefig('abduction.png')
        #     print("saved fig")
        #     quit()
        # if self.count == 248:
        #     plt.figure()
        #     plt.plot(np.arange(500), self.real[1], label="real actions")
        #     plt.plot(np.arange(500), self.pickle[1], label="sim actions")
        #     plt.legend()
        #     plt.savefig('hip.png')
        #     print("saved fig")
        # if self.count == 248:
        #     plt.figure()
        #     plt.plot(np.arange(500), self.real[2], label="real actions")
        #     plt.plot(np.arange(500), self.pickle[2], label="sim actions")
        #     plt.legend()
        #     plt.savefig('knee.png')
        #     print("saved fig")
        #     quit()

        # if self.count == 499:
        #     plt.plot(np.arange(500), self.real[0], label="real actions")
        #     plt.plot(np.arange(500), self.pickle[0], label="sim actions")
        #     plt.legend()
        #     plt.savefig('test.png')
        #     print("saved fig")
        #     quit()
        # if self.count == 248:
        #     plt.figure()
        #     plt.plot(np.arange(500), self.real[0], label="sim obs")
        #     plt.plot(np.arange(500), self.pickle[0], label="sim actions")
        #     plt.legend()
        #     plt.savefig('abduction.png')
        #     print("saved fig")
        #     quit()
        # if self.count == 248:
        #     plt.figure()
        #     plt.plot(np.arange(500), self.real[1], label="real actions")
        #     plt.plot(np.arange(500), self.pickle[1], label="sim actions")
        #     plt.legend()
        #     plt.savefig('hip.png')
        #     print("saved fig")
        # if self.count == 248:
        #     plt.figure()
        #     plt.plot(np.arange(500), self.real[2], label="real actions")
        #     plt.plot(np.arange(500), self.pickle[2], label="sim actions")
        #     plt.legend()
        #     plt.savefig('knee.png')
        #     print("saved fig")
        #     quit()

        # if self.count == 499:
        #     plt.plot(np.arange(500), self.real[0], label="real actions")
        #     plt.plot(np.arange(500), self.pickle[0], label="sim actions")
        #     plt.legend()
        #     plt.savefig('test.png')
        #     print("saved fig")
        #     quit()

        self.action = np.clip(action, -clip, clip)
        # print("real action: ", self.action)

        # self.real[:, self.count] = np.array(self.action)
        # self.pickle[:, self.count] = np.array(self.pickle_actions[self.count])

        # self.real[:, self.count] = np.array(self.pickle_obs[self.count][12:24])
        # self.pickle[:, self.count] = np.array(self.pickle_actions[self.count] * 0.15)

        joint_angles = (np.array(self.action) * 0.5 * 0.30) + self.reference_pose
        # print(self.pickle_obs[self.count][9:12])
        joint_angles[[0, 3, 6, 9]] = np.clip(joint_angles[[0, 3, 6, 9]], -0.1, 0.1)

        self._robot.Step(joint_angles)

        observations = self.get_observation()
        observations = torch.clip(observations, -clip, clip)

        self.pickle.append(observations)
        self.pickle_a.append(self.action)

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
        return self._robot.GetMotorAngles()

    def get_motor_velocities(self):
        return self._robot.GetMotorVelocities()
