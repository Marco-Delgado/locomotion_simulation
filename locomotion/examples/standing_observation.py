from absl import app
from absl import logging
import numpy as np
import time
from tqdm import tqdm
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client
import torch

from locomotion.robots import aliengo_robot
from locomotion.robots import robot_config

SWITCHED_POSITIONS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
FREQ = 0.5
default_dof_pos = np.array(
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


def get_observation(robot):
    observations = [
        robot.GetBaseVelocity(),
        robot.GetTrueBaseRollPitchYawRate(),
        robot.GetProjectedGravity(),
        robot.GetDirection(),
        robot.GetMotorAngles() - default_dof_pos,
        robot.GetMotorVelocities(),
    ]
    return observations


def main(_):
    logging.info("WARNING: this code executes low-level controller on the robot.")
    logging.info("Make sure the robot is hang on rack before proceeding.")
    input("Press enter to continue...")

    # Construct sim env and real robot
    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(numSolverIterations=30)
    p.setTimeStep(0.001)
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    robot = aliengo_robot.AliengoRobot(
        p,
        motor_control_mode=robot_config.MotorControlMode.HYBRID,
        enable_action_interpolation=False,
        reset_time=2,
        time_step=0.002,
        action_repeat=1,
    )

    # Move the motors slowly to initial position
    robot.ReceiveObservation()
    current_motor_angle = np.array(robot.GetMotorAngles())
    desired_motor_angle = np.array(
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
    for t in range(200):
        blend_ratio = np.minimum(t / 200.0, 1)
        action = (
            1 - blend_ratio
        ) * current_motor_angle + blend_ratio * desired_motor_angle
        robot.Step(action, robot_config.MotorControlMode.POSITION)

    for t in range(1000):
        action = desired_motor_angle
        robot.Step(action, robot_config.MotorControlMode.POSITION)
        obs = get_observation(robot)
        print(obs)

    robot.Terminate()


if __name__ == "__main__":
    app.run(main)
