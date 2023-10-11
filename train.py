import os
import argparse

from game_env import GameEnv
from players import Player, HeuristicPlayer

from stable_baselines3 import PPO, DDPG, A2C, TD3, SAC
from stable_baselines3.common.env_checker import check_env


def create_saving_directories():
    experiment_name = f"{args.fights_per_game}_fights_per_game"
    models_dir = os.path.join("models", experiment_name)
    logs_dir = os.path.join("logs", experiment_name)

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    return logs_dir, models_dir

def create_agent_name(args):
    return f"{args.algo}_{int(args.training_timesteps/1000)}k_steps_seed_{args.seed}"

def initialize_players():
    rl_player = Player(name='rl_player')
    bot_player = HeuristicPlayer(name='bot_player')
    return rl_player, bot_player

def initialize_game(args):
    rl_player.reset()
    bot_player.reset()
    env = GameEnv(rl_agent_player=rl_player, bot_player=bot_player, fights_per_game=args.fights_per_game, verbose=True)

    return env

algos = {"PPO": PPO,
         "DDPG": DDPG,
         "A2C": A2C,
         "TD3": TD3,
         "SAC": SAC}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, required=False, default="PPO")
    parser.add_argument('--fights_per_game', type=int, required=False, default=2)
    parser.add_argument('--nb_seeds', type=int, required=False, default=3)
    parser.add_argument('--total_timesteps', type=int, required=False, default=100000)

    args = parser.parse_args()

    # Creation of logs and models dir
    logs_dir, models_dir = create_saving_directories(args)
    agent_name = create_agent_name(args)
    
    # Initialization of the players
    rl_player, bot_player = initialize_players()
    
    # Checking of the environment
    env = GameEnv(rl_agent_player=rl_player, bot_player=bot_player, fights_per_game=args.fights_per_game, verbose=True)
    check_env(env, warn=True)

    # Train on multiple random seeds 
    for seed in range(args.nb_seeds):
        env = initialize_game(args)

        model = algos[args.algo]('MlpPolicy', env=env, verbose=1, tensorboard_log=logs_dir)

        model.learn(total_timesteps=args.training_timesteps, tb_log_name=agent_name)
        model.save(os.path.join(models_dir, agent_name))


