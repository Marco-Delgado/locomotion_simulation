#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
import traceback

import habitat
import numpy as np
import quaternion
from gym import Space, spaces
from habitat.config import Config as CN
from habitat.core.registry import registry
from habitat.core.simulator import (Config, Simulator)
from habitat.core.utils import try_cv2_import
from PIL import Image
from typing import Any

from locomotion.robots import a1_robot
from locomotion.robots import robot_config
from locomotion.robots import laikago_pose_utils
from locomotion.robots import a1
from locomotion.robots import a1_robot_velocity_estimator
from locomotion.robots import robot_config
from locomotion.envs import locomotion_gym_config
from robot_interface import RobotInterface


##uses spot api change to aliengo
#python api to control robot

cv2 = try_cv2_import()

def _aliengo_base_action_space():
    return spaces.Dict(
        {
            "vel_cmd": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        }
    )

ACTION_SPACES = {
    "ALIENGO": {
        "BASE_ACTIONS": _aliengo_base_action_space(),
    }
}

class Aliengo(object):
    r"""Simulator wrapper over PyRobot.
    PyRobot repo: https://github.com/facebookresearch/pyrobot
    To use this abstraction the user will have to setup PyRobot
    python3 version. Please refer to the PyRobot repository
    for setting it up. The user will also have to export a
    ROS_PATH environment variable to use this integration,
    please refer to `habitat.core.utils.try_cv2_import` for
    more details on this.
    This abstraction assumes that reality is a simulation
    (https://www.youtube.com/watch?v=tlTKTTt47WE).
    Args:
        config: configuration for initializing the PyRobot object.
    """

    def __init__(self, config):
        self._config = config['ALIENGO']
        
        # robot_sensors = []
        # for sensor_name in self._config.SENSORS:
        #     sensor_cfg = getattr(self._config, sensor_name)
        #     sensor_type = registry.get_sensor(sensor_cfg.TYPE)

        #     assert sensor_type is not None, "invalid sensor type {}".format(
        #         sensor_cfg.TYPE
        #     )
        #     robot_sensors.append(sensor_type(sensor_cfg))
        # self._sensor_suite = SensorSuite(robot_sensors)

        # config_pyrobot = {
        #     "base_controller": self._config.BASE_CONTROLLER,
        #     "base_planner": self._config.BASE_PLANNER,
        # }

        # assert (
        #     self._config.ROBOT in self._config.ROBOTS
        # ), "Invalid robot type {}".format(self._config.ROBOT)
        self._robot_config = getattr(self._config, self._config['ROBOT'].upper())

        action_spaces_dict = {}

        #TODO FIX
        self._action_space = self._robot_action_space(
            self._config['ROBOT'], self._robot_config
        )

        aliengo_config = self._robot_config
  
        #bosdyn.client.util.setup_logging(config.verbose)

        # # # # self.sdk = bosdyn.client.create_standard_sdk("Habitat2Spot")

        self._robot = a1_robot.A1Robot()
        self.verify_estop()

    def GetMotorAngles(self):
        return self._robot.GetMotorAngles(self)
















    def verify_estop(self):
        """Verify the robot is not estopped"""
        robot = self._robot
        client = robot.ensure_client(EstopClient.default_service_name)
        if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
            error_message = (
                "Robot is estopped. Please use an external E-Stop client, such as the"
                " estop SDK example, to configure E-Stop."
            )
            robot.logger.error(error_message)
            raise Exception(error_message)

    def _robot_action_space(self, robot_type, robot_config):
        action_spaces_dict = {}
        for action in robot_config.ACTIONS:
            action_spaces_dict[action] = ACTION_SPACES[robot_type.upper()][
                action
            ]
        return spaces.Dict(action_spaces_dict)

    @property
    def action_space(self):
        return self._action_space

    def power_off(self):
        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        self._robot
        self._robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not self._robot.is_powered_on(), "Robot power off failed."
        self._robot.logger.info("Robot safely powered off.")

    def step(self, action, action_params):
        r"""Step in reality. 
        """
        robot = self._robot
        robot.Step(self, action, action_params)

        step_time = time.time()
        vel_end_time = 0.2
        x_vel, y_vel, z_vel, ang_vel = action_params['xyt_vel']

        vision_tform_body = get_vision_tform_body(
                    self.state_client.get_robot_state().kinematic_state.transforms_snapshot
                )
        body_tform_goal = math_helpers.SE2Velocity(
            x=x_vel, y=y_vel, angular=ang_vel
        )
        robot_cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=body_tform_goal.linear_velocity_x,
            v_y=body_tform_goal.linear_velocity_y,
            v_rot=body_tform_goal.angular_velocity,
            frame_name=BODY_FRAME_NAME,
            params=self.obstacle_params
        )
        start_time = time.time()
        self.command_client.robot_command(
            lease=None, command=robot_cmd, end_time_secs=time.time() + vel_end_time
        )
        while not time.time() > start_time+ vel_end_time:
            pass
        agent_state = self.get_agent_state()
        base_state = agent_state["base"]
        vel = agent_state["vel"]
        print("Actual Position: ", base_state)
        print("Actual Velocity: ", vel)
        print('Total Time: ', time.time() - step_time, 'Command Time: ', vel_end_time)

    def get_agent_state(
        self, agent_id: int = 0, base_state_type: str = "odom"
    ):
        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )
        agent_kinematic_state_bos = self.state_client.get_robot_state().kinematic_state
        agent_state_bos = get_vision_tform_body(agent_kinematic_state_bos.transforms_snapshot)
        agent_state_bos_vel_vis = agent_kinematic_state_bos.velocity_of_body_in_vision
        rot = agent_state_bos.rotation
        position = [agent_state_bos.x, agent_state_bos.y, agent_state_bos.z]
        rot_quats = [rot.w, rot.x, rot.y, rot.z]
        as_angles = math_helpers.quat_to_eulerZYX(rot)
        yaw = as_angles[0]

        state = {
            "base" : position[:2] + [yaw],
            "pos" : position,
            "quat" : rot_quats,
            "rpy" : as_angles,
            "vel": [agent_state_bos_vel_vis.linear.x, agent_state_bos_vel_vis.linear.y, agent_state_bos_vel_vis.angular.z]
        }
        return state

    def seed(self, seed):
        raise NotImplementedError("No support for seeding in reality")
    
    def close(self):
        self.power_off()
        self.lease_alive.shutdown()
