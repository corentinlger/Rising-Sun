import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DDPG, A2C, TD3, SAC
from stable_baselines3.common.env_checker import check_env

from game.game_env import GameEnv, create_agent_name, initialize_players, initialize_game
from game.players import Player, HeuristicPlayer, SepukuPoetsPlayer, HumanPlayer, bot_player_dict, load_policy
from utils.algos import algos
from utils.utils import create_saving_directories, plot_evaluation_results


# TODO : Add functions to test the desired model with arguments in the python file (algorithm, training timesteps ... )

def load_policy(algo, algos, models_dir, agent_name, seed):
    """
    Load a trained RL policy
    :param algo : (str) The algorithm used
    :param models : {(str)} Hashmap of supported stable-baselines3 algorithms
    :param models_dir : (str) Path to the models saving directory
    :param agent_name : (str) Name of the RL agent in models directory
    :param seed : (int) Random seed during the training 
    """
    model_name = os.path.join(models_dir, agent_name)
    try:
        model = algos[algo].load(model_name)
    except:
        raise(ValueError(f"No model has been trained with this configuration yet"))
    return model

def evaluate_model(args, env, model):
    """
    Evaluate a trained RL agent against another bot
    :param args: TODO : replace by specific args
    :param env : (GameEnv) Gymnasium environment
    :param model : {(int)} Neural Network of the agent's  action policy
    """

    rl_player_wins, rl_player_wins_list, rl_player_nb_points = 0, [], []
    bot_player_wins, bot_player_wins_list, bot_player_nb_points = 0, [], []

    for game_nb in range(args.nb_testing_games):
        obs, info = env.reset()
        done = False
        rl_ep_nb_points = []
        bot_ep_nb_points = []
        while not done:
            print("")
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            rl_ep_nb_points.append(rl_player.nb_points)
            bot_ep_nb_points.append(bot_player.nb_points)
            print(f"rl_player points : {rl_player.nb_points} bot_player points : {bot_player.nb_points}")

        # If episode is finished
        rl_player_nb_points.append(np.mean(rl_ep_nb_points))
        bot_player_nb_points.append(np.mean(bot_ep_nb_points))
        if rl_player.nb_points > bot_player.nb_points:
            rl_player_wins += 1
        else:
            bot_player_wins += 1
        rl_player_wins_list.append(rl_player_wins)
        bot_player_wins_list.append(bot_player_wins)

    mean_rl_ep_points = np.mean(rl_player_nb_points)
    mean_bot_ep_points = np.mean(bot_player_nb_points)

    print("")
    print(f"mean rl_player points : {mean_rl_ep_points}  mean bot_player points : {mean_bot_ep_points}")
    return rl_player_wins_list, bot_player_wins_list



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, required=False, default="PPO")
    parser.add_argument('--fights_per_game', type=int, required=False, default=2)
    parser.add_argument("--bot_behavior", type=str, required=False, default="random")
    parser.add_argument('--nb_seeds', type=int, required=False, default=3)
    parser.add_argument('--total_timesteps', type=int, required=False, default=100000)
    parser.add_argument('--nb_testing_games', type=int, required=False, default=1000)

    args = parser.parse_args()

    logs_dir, models_dir = create_saving_directories(args)

    agent_name = create_agent_name(args)
    model = load_policy(args, agent_name, models_dir)  

    rl_player, bot_player = initialize_players(args.bot_behavior)
    env = initialize_game(args, rl_player, bot_player) 
    check_env(env, warn=True)

    rl_player_wins_list, bot_player_wins_list = evaluate_model(args, env, model)
    plot_evaluation_results(rl_player_wins_list, bot_player_wins_list)



