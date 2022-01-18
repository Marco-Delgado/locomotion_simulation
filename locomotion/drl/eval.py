from locomotion.a2c_ppo_acktr import model

from absl import app
from absl import logging
import numpy as np
import time
import pybullet as p  # pytype: disable=import-error

from locomotion.envs import locomotion_gym_config
from locomotion.envs import env_builder
from locomotion.robots import a1
from locomotion.robots import a1_robot
from locomotion.robots import robot_config
from tqdm import tqdm
from locomotion.envs import locomotion_gym_env

import argparse
import torch

sim_params = locomotion_gym_config.SimulationParameters()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to .pth file")
    parser.add_argument("-r", "--real_robot", default=False, action="store_true")
    # parser.add_argument(
    #     "opts",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    #     help="Modify config options from command line",
    # )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))

    """ Generate config """
    config = checkpoint["config"]
    # config.merge_from_list(args.opts)
    # config.defrost()

    """ Create environment """
    if args.real_robot:
        env = locomotion_gym_env.LocomotionGymEnv(
        gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params),
        robot_class=a1_robot.A1Robot,
        is_render=False,
        on_rack=True
        )
    else:
        env = locomotion_gym_env.LocomotionGymEnv(
        gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params),
        robot_class=a1.A1,
        is_render=False,
        on_rack=False
        )

    """ Create actor-critic """
    num_reward_terms = checkpoint["state_dict"][
        "actor_critic.base.critic_linear.bias"
    ].shape[0] - 1
    actor_critic = model.Policy(
        env.observation_space.shape,
        env.action_space,
        base_kwargs={
            "recurrent": config.RECURRENT_POLICY,
            "reward_terms": num_reward_terms,
            "hidden_size": config.RL.PPO.hidden_size,
            "mlp_hidden_sizes": config.RL.PPO.mlp_hidden_sizes,
        },
    )
    print("\nActor-critic architecture:")
    device = torch.device("cpu")
    actor_critic.to(device)

    """ Load weights """
    actor_critic.load_state_dict(
        {
            k[len("actor_critic."):]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("actor_critic")
        }
    )

    """ Execute episodes """
    num_episodes = 1
    for idx in range(num_episodes):
        observations = env.reset()
        recurrent_hidden_states = torch.zeros(
            1,
            actor_critic.base.recurrent_hidden_state_size,
            config.RL.PPO.hidden_size,
            device=device,
        )
        not_done = torch.ones(1, 1).to(device)
        step_count = 0
        while not_done[0]:
            start_time = time.time()
            step_count += 1
            observations[-4:] = np.array([1., 1., 1., 1.])
            (
                _,
                action,
                _,
                recurrent_hidden_states,
            ) = actor_critic.act(
                torch.FloatTensor(observations).unsqueeze(0).to(device),
                recurrent_hidden_states,
                not_done,
                deterministic=True
            )

            print(action)
            observations, _, done, _ = env.step(action.flatten().detach().cpu().numpy())
            # time.sleep(1/30 - 0.02)
            print("time: ", time.time() - start_time, 1 / (time.time() - start_time))
            not_done[0] = not done
            # while time.time() < start_time + 1 / 30:
            #     pass

        print(f'Episode #{idx + 1} finished in {step_count} steps')