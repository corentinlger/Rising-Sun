import os

from GameEnv_rllib import GameEnv
from players import Player, HeuristicPlayer

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import EnvContext


if __name__ == '__main__':
    print(ray.__version__)

    # Parameters of the game
    fights_per_game = 2
    rl_player = Player(name='rl_player')
    bot_player = HeuristicPlayer(name='bot_player')

    env_config = {"bot_player": rl_player,
                  "rl_agent_player": rl_player,
                  "fights_per_game": fights_per_game,
                  "verbose": False}

    config = PPOConfig().environment(env=GameEnv, env_config=env_config).training(train_batch_size=4000)

    print("Training start")

    # `storage_path' instead of local_dir
    # trains 4000 by 4000 steps
    tune.run(
        "PPO",
        name="RLlib_test_RS",
        stop={"timesteps_total": 100_000},
        config=config,
        local_dir="logs",
        checkpoint_freq=1,
        resume=False
    )

    print("Training ended")



