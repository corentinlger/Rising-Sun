import os
import argparse

from stable_baselines3 import PPO, DDPG, A2C, TD3, SAC
from stable_baselines3.common.env_checker import check_env

from game.game_env import initialize_players, initialize_game
from game.players import create_agent_name
from utils.utils import create_saving_directories
from utils.algos import algos

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, required=False, default="")
    parser.add_argument('--algo', type=str, required=False, default="PPO")
    parser.add_argument('--fights_per_game', type=int, required=False, default=2)
    parser.add_argument("--bot_behavior", type=str, required=False, default="random")
    parser.add_argument('--nb_seeds', type=int, required=False, default=3)
    parser.add_argument('--total_timesteps', type=int, required=False, default=100000)
    parser.add_argument('--verbose', type=str, required=False, default="True")

    args = parser.parse_args()

    logs_dir, models_dir = create_saving_directories(args.exp_name, args.fights_per_game, args.bot_behavior)
    agent_name = create_agent_name(args.algo, args.total_timesteps)
    
    rl_player, bot_player = initialize_players(args.bot_behavior)
    
    # Checking of the environment
    env = initialize_game(rl_player, bot_player, args.fights_per_game)
    check_env(env, warn=True)

    for seed in range(args.nb_seeds):
        env = initialize_game(rl_player, bot_player, args.fights_per_game, verbose=False)

        model = algos[args.algo]('MlpPolicy', env=env, verbose=1, tensorboard_log=logs_dir)

        model.learn(total_timesteps=args.total_timesteps, tb_log_name=f"{agent_name}_seed_{seed}")
        model.save(os.path.join(models_dir, f"{agent_name}__seed_{seed}"))


