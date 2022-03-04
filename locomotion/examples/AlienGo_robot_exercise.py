"""Apply the same action to the simulated and real A1 robot.


As a basic debug tool, this script allows you to execute the same action
(which you choose from the pybullet GUI) on the simulation and real robot
simultaneouly. Make sure to put the real robbot on rack before testing.
"""

from absl import app
from absl import logging
import numpy as np
import time
from tqdm import tqdm
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client

from locomotion.robots import aliengo
from locomotion.robots import aliengo_robot
from locomotion.robots import robot_config
from locomotion.robots import a1_robot_velocity_estimator

FREQ = 0.5


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
    desired_motor_angle = np.array([0.0, 0.9, -1.8] * 4)
    for t in tqdm(range(600)):
        blend_ratio = np.minimum(t / 200.0, 1)
        action = (
            1 - blend_ratio
        ) * current_motor_angle + blend_ratio * desired_motor_angle
        robot.Step(action, robot_config.MotorControlMode.POSITION)
        velocity_estimate = a1_robot_velocity_estimator.VelocityEstimator(robot)
        print(velocity_estimate.estimated_velocity())
        time.sleep(0.005)

    # Move the legs in a sinusoidal curve
    for t in tqdm(range(1000)):
        angle_hip = 0.9 + 0.2 * np.sin(2 * np.pi * FREQ * 0.01 * t)
        angle_calf = -2 * angle_hip
        action = np.array([0.0, angle_hip, angle_calf] * 4)
        robot.Step(action, robot_config.MotorControlMode.POSITION)
        time.sleep(0.007)

    robot.Terminate()


if __name__ == "__main__":
    app.run(main)
