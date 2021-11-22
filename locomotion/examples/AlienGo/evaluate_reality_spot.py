import argparse
import random
import time
from collections import OrderedDict, defaultdict

from omegaconf import OmegaConf
import hydra
import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat.sims import make_sim
from PIL import Image
from spot_config import get_spot_config
from spot_sim import *
import utils
sys.path.insert(0, '/home/spot/camera_ws/src/spot_ros/src')
from camera import collect_images
import rospy

DEVICE = torch.device("cpu")

LOG_FILENAME = "exp.navigation.log"
MAX_DEPTH = 10.0

class NavEnv:
    def __init__(self):
        config = get_spot_config()

        # self._reality = make_sim(id_sim="Spot-v0", config=config.SPOT)
        self._reality = Spot(config)

        print(self._reality)
        self._pointgoal_key = "pointgoal_with_gps_compass"

    # def _pointgoal(self, obs):
    #     agent_x, agent_y, agent_rotation = obs['base']
    #     agent_coordinates = np.array([agent_x, agent_y])
    #     rho = np.linalg.norm(agent_coordinates - obs['target'])
    #     theta = (
    #         np.arctan2(
    #             obs['target'][1] - agent_coordinates[1], obs['target'][0] - agent_coordinates[0]
    #         )
    #         - agent_rotation
    #     )
    #     theta = theta % (2 * np.pi)
    #     if theta >= np.pi:
    #         theta = -((2 * np.pi) - theta)
    #     return rho, theta

    def _get_base_state(self):
        state = self._reality.get_agent_state()
        base_state = np.array(state['base'], dtype=np.float32)
        print('base_state: ', base_state)
        return base_state

    def _get_pos_rpy(self):
        state = self._reality.get_agent_state()
        pos = np.array(state['pos'], dtype=np.float32)
        rpy = np.array(state['rpy'], dtype=np.float32)
        return pos, rpy

    def _get_vel(self):
        state = self._reality.get_agent_state()
        vel = np.array(state['vel'], dtype=np.float32)
        return vel

    def _get_obs(self):
        return self._camera.get_depth()

    def _get_z(self, agent):
        return agent.z

    def reset(self):
        return self._reality.reset()

    @property
    def reality(self):
        return self._reality

    def step(self, action):
        observations = self._reality.step(
            "vel_cmd",
            {
                "xyt_vel": action,
            },
        )

        return observations

    def close(self) -> None:
        self._reality.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def log_mesg(mesg):
    print(mesg)
    with open(LOG_FILENAME, "a") as f:
        f.write(mesg + "\n")

def load_model(cfg,policy_dir, ckpt):
    pass

def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)

def get_high_level_command(obs, goal, model):
    state = {}
    frames = []
    state['sensor'] = goal
    obs['depth'] = np.reshape(obs['depth'], [1, obs['depth'].shape[0], obs['depth'].shape[1]])
    obs['depth'] = (obs['depth'] * 255).round().astype(np.uint8)
    for _ in range(3):
        frames.append(obs['depth'])
    obs['depth'] = np.concatenate(list(frames), axis=0)
    print('obs_depth_shape: ', obs['depth'].shape)
    state['depth'] = obs['depth']

    action = model.act(state, model.a1_z_in, sample=False)

    return [action[0], action[1], 0.0], action[2]

@hydra.main(config_path="config/base_config.yaml", strict=True)
def main(cfg):
    # self._reality.obstacle_avoidance = args.obstacle_avoidance
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print("Starting new episode")

    with NavEnv() as env:
        env.reset()
        obs = {}
        start_pos = env._get_base_state()
        goal_list = [cfg.goal_x + start_pos[0], cfg.goal_y + start_pos[1]]
        obs['target'] = np.array(goal_list, dtype=np.float32)

        num_actions = 0
        reached_goal = False
        lin_speed = [0.0, 0.0, 0.0]
        ang_speed = 0.0

        model = load_model(cfg,
            policy_dir=cfg.policies_dir,
            ckpt=cfg.ckpt,
        )
        time.sleep(30)
        # while True:
        #     obs['pos'], obs['rpy'] = env._get_pos_rpy()
        #     obs['base'] = env._get_base_state()
        #     obs['depth'] = listener.curr_depth

        #     goal = env._pointgoal(obs)

        #     print(
        #         "Your goal is to get to: {:.3f}, {:.3f}  "
        #         "rad ({:.2f} degrees)".format(
        #             goal[0], goal[1], (goal[1] / np.pi) * 180
        #         )
        #     )

        #     # goal = utils.global_to_local(obs['target'], obs['pos'], obs['rpy'])[:2]
        #     # goal = np.array(utils.cartesian_to_polar(goal[0], goal[1]))

        #     if np.linalg.norm(obs['pos'][:2] - obs['target'][:2]) < 0.35 or reached_goal:
        #       lin_speed = [0.0, 0.0, 0.0]
        #       ang_speed = 0.0
        #       speed = np.array([lin_speed[0], lin_speed[1], lin_speed[2], ang_speed], dtype=np.float32)
        #       env.step(speed)
        #       reached_goal = True
        #       print("STOP called, episode over.")
        #       print("Distance to goal: {:.3f}m".format(goal[0]))
        #       return
        #     else:
        #       lin_speed, ang_speed = get_high_level_command(obs, goal, model)
        #       speed = np.array([lin_speed[0], lin_speed[1], lin_speed[2], ang_speed], dtype=np.float32)
        #       print("target speed = ", speed)
        #       env.step(speed)
        #       num_actions += 1
#            input("Press key to continue")

rospy.init_node('listener', anonymous=True)
listener = collect_images()

if __name__ == "__main__":
    main()
