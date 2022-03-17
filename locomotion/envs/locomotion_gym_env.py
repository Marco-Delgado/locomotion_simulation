# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file implements the locomotion gym env."""
import collections
import time
import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np
import torch
import pybullet  # pytype: disable=import-error
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd

from typing import List, Tuple, Union
from habitat.utils.geometry_utils import wrap_heading
from locomotion.robots import robot_config
from locomotion.envs.sensors import sensor
from locomotion.envs.sensors import space_utils

SWITCHED_POSITIONS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


class LocomotionGymEnv(gym.Env):
    def __init__(self, gym_config, robot_class=None, is_render=False, on_rack=False):
        self.new_joint_angles = [0, 0, 0] * 4
        self._last_true_motor_angle = None
        self.observation_space = Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(30,),
            dtype=np.float32,
        )

        self.action_space = Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(12,),
            dtype=np.float32,
        )
        self._gym_config = gym_config
        self._robot_class = robot_class
        self._is_render = is_render
        self._on_rack = on_rack

        self.joint_limits_lower = np.array([-0.1, -np.pi / 3, -5 / 6 * np.pi] * 4)
        self.joint_limits_upper = np.array([0.1, np.pi / 2.1, -np.pi / 4] * 4)
        self.deafult_dof_pos = np.array(
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

        # Checks to see if the environment should be rendered or applied in the real world
        if self._is_render:
            self._pybullet_client = bullet_client.BulletClient(
                connection_mode=pybullet.GUI
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI,
                gym_config.simulation_parameters.enable_rendering_gui,
            )
        else:
            self._pybullet_client = bullet_client.BulletClient(
                connection_mode=pybullet.DIRECT
            )
        self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())

        # TODO: check if this can be placed inside "if self.is_render"
        if gym_config.simulation_parameters.egl_rendering:
            self._pybullet_client.loadPlugin("eglRendererPlugin")

        self._robot = self._robot_class(
            pybullet_client=self._pybullet_client,
            on_rack=self._on_rack,
            motor_control_mode=self._gym_config.simulation_parameters.motor_control_mode,
        )

        self.reset()

    def reset(
        self,
        initial_motor_angles=None,
        reset_duration=0.0,
    ):
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(300 / 10)
        self._pybullet_client.setTimeStep(0.001)
        self._pybullet_client.setGravity(0, 0, -10)
        self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        self._pybullet_client.loadURDF("plane.urdf")

        self._robot.Reset(
            reload_urdf=True,
            default_motor_angles=initial_motor_angles,
            reset_time=reset_duration,
        )

        self._last_true_motor_angle = np.array(self._robot.GetMotorAngles())
        observations = self._get_observation()
        # observations = np.concatenate(list(observations.values()))
        return observations

    def step(self, action):
        action = np.array(action)

        for i, d in enumerate(action):
            if abs(d) <= 0.1:
                action[i] = 0
        # Clip actions and scale
        delta = np.clip(action, -1.0, 1.0) * np.deg2rad(10)

        self._last_true_motor_angle = np.array(self._robot.GetMotorAngles())

        self.new_joint_angles = np.clip(
            wrap_heading(self._last_true_motor_angle + delta),
            self.joint_limits_lower[:],
            self.joint_limits_upper[:],
        )

        self._robot.Step(self.new_joint_angles, robot_config.MotorControlMode.POSITION)

        observations = self._get_observation()
        observations = np.concatenate(list(observations.values()))

        return observations, 0, False, {}

    def _get_observation(self):
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
                    self._robot.GetMotorAngles() - self.deafult_dof_pos,
                    dtype=torch.float,
                ),
                torch.tensor(self._robot.GetMotorVelocities(), dtype=torch.float),
                torch.tensor(self.new_joint_angles, dtype=torch.float),
            ),
            dim=-1,
        )
        return observations
