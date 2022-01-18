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

    self.observation_space = spaces.Box(
      low=np.finfo(np.float32).min,
      high=np.finfo(np.float32).max,
      shape=(31,),
      dtype=np.float32,
    )

    self.action_space = spaces.Box(
      low=np.finfo(np.float32).min,
      high=np.finfo(np.float32).max,
      shape=(12,),
      dtype=np.float32,
    )

    if self._is_render:
      self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
      pybullet.configureDebugVisualizer(
        pybullet.COV_ENABLE_GUI,
        gym_config.simulation_parameters.enable_rendering_gui)
    else:
      self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())
    if gym_config.simulation_parameters.egl_rendering:
      self._pybullet_client.loadPlugin('eglRendererPlugin')

    self._robot = self._robot_class(
      pybullet_client=self._pybullet_client,
      on_rack=self._on_rack,
      motor_control_mode=self._gym_config.simulation_parameters.motor_control_mode,
    )

    self.reset()


  def reset(self,
            initial_motor_angles=np.array([0,1.17,-2.77]*4),
            reset_duration=1.0, ):
    self._pybullet_client.resetSimulation()
    self._pybullet_client.setPhysicsEngineParameter(300 / 10)
    self._pybullet_client.setTimeStep(0.001)
    self._pybullet_client.setGravity(0, 0, -10)

    self._world_dict = {
      "ground": self._pybullet_client.loadURDF("plane_implicit.urdf")
    }

    self._robot.Reset(reload_urdf=False,
                      default_motor_angles=initial_motor_angles,
                      reset_time=reset_duration)

    return self._get_observation()

  def step(self, action):
    action = action[SWITCHED_POSITIONS]

    observations = self._get_observation()

    deltas = []
    # for i, j in zip(observations[:12], np.array([0, 0.432, -0.77] * 4)):
    for i, j in zip(observations[:12], np.array([0, 0.9, -1.8] * 4)):
      err = i - j
      # Be more precise than necessary
      if abs(err) > np.deg2rad(10) / 3:
        # Flip direction based on error
        coeff = 1 if err < 0 else -1
        deltas.append(coeff * min(np.deg2rad(10), abs(err)))
      else:
        deltas.append(0)

    deltas = np.array(deltas, dtype=np.float32)

    # print("actions: ", action)
    # deltas = np.clip(action, -1.0, 1.0) * np.deg2rad(5)
    desired_motor_angle = deltas + observations[:12]
    # print("observations: ", observations)

    # self._robot.Step(desired_motor_angle, robot_config.MotorControlMode.POSITION)
    # self._robot.Step(desired_motor_angle)
    low_timeout = time.time() + 1 / 30
    count = 0
    while time.time() < low_timeout:
      curr_joints = self._get_observation()[:12]
      deltas = []
      for i, j in zip(curr_joints, desired_motor_angle):
        err = i - j
        # Be more precise than necessary
        if abs(err) > np.deg2rad(10) / 3:
          # Flip direction based on error
          coeff = 1 if err < 0 else -1
          deltas.append(coeff * min(np.deg2rad(10), abs(err)))
        else:
          deltas.append(0)
      blend_desired_motor_angle = np.array(deltas, dtype=np.float32) + curr_joints
      self._robot.Step(blend_desired_motor_angle)
      count += 1
      print(count)

    return observations, 0, False, {}

  def _get_observation(self):
    # observations = \
    #   {
    #     "joint_pos": self._robot.GetMotorAngles()[SWITCHED_POSITIONS],
    #     "joint_vel": self._robot.GetMotorVelocities()[SWITCHED_POSITIONS],
    #     "euler_rot": self._robot.GetBaseRollPitchYaw(),
    #     "feet_contact": self._robot.GetFootContacts()
    #   }

    observations = np.array([*self._robot.GetMotorAngles()[SWITCHED_POSITIONS], *self._robot.GetMotorVelocities()[SWITCHED_POSITIONS], *self._robot.GetBaseRollPitchYaw(), *self._robot.GetFootContacts()])

    return observations
