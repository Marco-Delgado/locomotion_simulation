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
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet  # pytype: disable=import-error
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd

from locomotion.robots import robot_config
from locomotion.envs.sensors import sensor
from locomotion.envs.sensors import space_utils

SWITCHED_POSITIONS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


class LocomotionGymEnv(gym.Env):
  def __init__(self,
               gym_config,
               robot_class=None,
               is_render=False,
               on_rack=False
               ):

    self._gym_config = gym_config
    self._robot_class = robot_class
    self._is_render = is_render
    self._on_rack = on_rack

    # Checks to see if the environment should be rendered or applied in the real world
    if self._is_render:
      self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
      pybullet.configureDebugVisualizer(
        pybullet.COV_ENABLE_GUI,
        gym_config.simulation_parameters.enable_rendering_gui)
    else:
      self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())

    # TODO: check if this can be placed inside "if self.is_render"
    if gym_config.simulation_parameters.egl_rendering:
      self._pybullet_client.loadPlugin('eglRendererPlugin')

    self.reset()

  def reset(self,
            initial_motor_angles=None,
            reset_duration=0.0, ):
    self._pybullet_client.resetSimulation()
    self._pybullet_client.setPhysicsEngineParameter(300 / 10)
    self._pybullet_client.setTimeStep(0.001)
    self._pybullet_client.setGravity(0, 0, -10)

    self._robot = self._robot_class(
      pybullet_client=self._pybullet_client,
      on_rack=self._on_rack,
      motor_control_mode=self._gym_config.simulation_parameters.motor_control_mode,
    )

    self._robot.Reset(reload_urdf=False,
                      default_motor_angles=initial_motor_angles,
                      reset_time=reset_duration)

    return self._get_observation()

  def step(self, action):
    action = action[SWITCHED_POSITIONS]
    self._robot.Step(action)

    return self._get_observation(), 0, False, {}

  def _get_observation(self):
    observations = \
      {
        "joint_pos": self._robot.GetMotorAngles()[SWITCHED_POSITIONS],
        "joint_vel": self._robot.GetMotorVelocities()[SWITCHED_POSITIONS],
        "euler_rot": self._robot.GetBaseRollPitchYaw(),
        "feet_contact": self._robot.GetFootContacts()
      }

    return observations
