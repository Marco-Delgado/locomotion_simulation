"""Apply the same action to the simulated and real A1 robot.


As a basic debug tool, this script allows you to execute the same action
(which you choose from the pybullet GUI) on the simulation and real robot
simultaneouly. Make sure to put the real robbot on rack before testing.
"""

from absl import app
from absl import logging
import numpy as np
import pybullet as p  # pytype: disable=import-error

from locomotion.envs import locomotion_gym_config
from locomotion.envs import env_builder
from locomotion.robots import a1
from locomotion.robots import a1_robot
from locomotion.robots import robot_config
from tqdm import tqdm
from locomotion.envs import locomotion_gym_env

sim_params = locomotion_gym_config.SimulationParameters()


def main(_):
  logging.info("WARNING: this code executes low-level controller on the robot.")
  logging.info("Make sure the robot is hang on rack before proceeding.")
  input("Press enter to continue...")

  # Construct sim env and real robot
  sim_env = locomotion_gym_env.LocomotionGymEnv(
    gym_config=locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params),
    robot_class=a1.A1,
    is_render=True,
    on_rack=True
  )
  real_env = locomotion_gym_env.LocomotionGymEnv(
    gym_config=locomotion_gym_config.LocomotionGymConfig(
      simulation_parameters=sim_params),
    robot_class=a1_robot.A1Robot,
    is_render=False,
    on_rack=True
  )

  # The code slowly stands and the moves the limbs up and down (similar to a push up)
  current_motor_angle = np.array(real_env._get_observation()["joint_pos"])
  desired_motor_angle = np.array([0., 0.9, -1.8] * 4)

  # Sets the robot to stand using blending
  for t in tqdm(range(300)):
    blend_ratio = np.minimum(t / 200., 1)
    action = (1 - blend_ratio
              ) * current_motor_angle + blend_ratio * desired_motor_angle

    real_env.step(action)
    sim_env.step(action)

  # Sets the robot to move it's limbs up and down (push up)
  for t in tqdm(range(1000)):
    angle_hip = 0.9 + 0.2 * np.sin(2 * np.pi * 0.5 * 0.01 * t)
    angle_calf = -2 * angle_hip
    action = np.array([0., angle_hip, angle_calf, 0., 0.9, -1.8, 0., 0.9, -1.8, 0., 0.9, -1.8])

    real_env.step(action)
    sim_env.step(action)


if __name__ == '__main__':
  app.run(main)
