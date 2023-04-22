import os
import numpy as np
import tensorboard
from GameEnv import GameEnv
from Players import Player, HeuristicPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

if __name__ == '__main__':

    # Parameters of the game
    fights_per_game = 2
    nb_seeds = 3
    total_timesteps = 200_000

    # Creation of logs and models dir
    log_dir = os.path.abspath("logs")
    model_dir = os.path.abspath("models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Initialization of the players
    rl_player = Player(name='rl_player')
    bot_player = HeuristicPlayer(name='bot_player')

    # Checking of the environment
    env = GameEnv(rl_agent_player=rl_player, bot_player=bot_player, fights_per_game=fights_per_game)
    check_env(env, warn=True)

    for seed in range(nb_seeds):
        rl_player.reset()
        bot_player.reset()
        env = GameEnv(rl_player, bot_player, fights_per_game)

        model = PPO('MlpPolicy', env=env, verbose=1, tensorboard_log=log_dir)
        model.learn(total_timesteps, reset_num_timesteps=False, tb_log_name=f"{log_dir}/seed_{seed}_{int(total_timesteps/1000)}k_steps")
        model.save(f"{model_dir}/seed_{seed}_{int(total_timesteps/1000)}k_steps")


