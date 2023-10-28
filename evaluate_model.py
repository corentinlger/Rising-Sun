import os
import argparse

import numpy as np

from stable_baselines3.common.env_checker import check_env

from game.game_env import initialize_players, initialize_game
from game.players import Player, TrainedPlayer
from utils.utils import create_saving_directories, plot_evaluation_results


def evaluate_model(nb_testing_games, env, model):
    """
    Evaluate a trained RL agent against another bot
    :param nb_testing_games: (int) Number of games where agents compete with each other
    :param env : (GameEnv) Gymnasium environment
    :param model : {(int)} Neural Network of the agent's  action policy
    """

    rl_player_wins, rl_player_wins_list, rl_player_nb_points = 0, [], []
    bot_player_wins, bot_player_wins_list, bot_player_nb_points = 0, [], []

    for _ in range(nb_testing_games):
        obs, info = env.reset()
        done = False
        rl_ep_nb_points = []
        bot_ep_nb_points = []
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            rl_ep_nb_points.append(rl_player.nb_points)
            bot_ep_nb_points.append(bot_player.nb_points)
            # print(f"rl_player points : {rl_player.nb_points} bot_player points : {bot_player.nb_points}")

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

    print(f"\nmean rl_player points : {mean_rl_ep_points}  mean bot_player points : {mean_bot_ep_points}")
    return rl_player_wins_list, bot_player_wins_list



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--exp_name', type=str, required=False, default="")
    parser.add_argument('--algo', type=str, required=False, default="PPO")
    parser.add_argument('--fights_per_game', type=int, required=False, default=2)
    parser.add_argument("--bot_behavior", type=str, required=False, default="random")
    parser.add_argument("--tr_bot_behavior", type=str, required=False, default="random")
    parser.add_argument('--tr_timesteps', type=int, required=False, default=100000)
    parser.add_argument('--nb_seeds', type=int, required=False, default=3)
    parser.add_argument('--seed', type=int, required=False, default=0)
    parser.add_argument('--nb_testing_games', type=int, required=False, default=1000)

    args = parser.parse_args()

    logs_dir, models_dir = create_saving_directories(args.exp_name, args.fights_per_game, args.tr_bot_behavior)

    player = Player(args.fights_per_game)
    trained_player = TrainedPlayer(exp_name=args.exp_name,
                                   fights_per_game=args.fights_per_game, 
                                   algo=args.algo, 
                                   tr_timesteps=args.tr_timesteps,
                                   tr_bot_behavior=args.tr_bot_behavior, 
                                   seed=args.seed)
    model = trained_player.policy 
    
    rl_player, bot_player = initialize_players(args.bot_behavior, trained_player)
    env = initialize_game(player=player, bot_player=bot_player, fights_per_game=args.fights_per_game, verbose=False) 
    check_env(env, warn=True)


    rl_player_wins_list, bot_player_wins_list = evaluate_model(args.nb_testing_games, env, model)
    plot_evaluation_results(rl_player_wins_list, bot_player_wins_list)



