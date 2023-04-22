import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from GameEnv import GameEnv
from Players import Player, HeuristicPlayer

# Trained model loading
model = PPO.load('models/seed_0_200k_steps')

# Players and Env creation
rl_player = Player('rl_player')
bot_player = HeuristicPlayer('bot_player')
env = GameEnv(rl_player, bot_player)

# Testing parameters
nb_testing_games = 1000

# Creation of lists to score the victories
rl_player_wins, rl_player_wins_list, rl_player_nb_points = 0, [], []
bot_player_wins, bot_player_wins_list, bot_player_nb_points = 0, [], []

for game_nb in range(nb_testing_games):
    obs = env.reset()
    done = False
    rl_ep_nb_points = []
    bot_ep_nb_points = []
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rl_ep_nb_points.append(rl_player.nb_points)
        bot_ep_nb_points.append(bot_player.nb_points)
        print(f"rl_player points : {rl_player.nb_points} bot_player points : {bot_player.nb_points}")
        print(f"action : {action} nb_gold_action : {np.sum(action)}/{rl_player.golds}")

    # If episode is finished
    rl_player_nb_points.append(np.mean(rl_ep_nb_points))
    rl_player_nb_points.append(np.mean(rl_ep_nb_points))
    if rl_player.nb_points > bot_player.nb_points:
        rl_player_wins += 1
    else:
        bot_player_wins += 1
    rl_player_wins_list.append(rl_player_wins)
    bot_player_wins_list.append(bot_player_wins)

mean_rl_ep_points = np.mean(rl_player_nb_points)
mean_bot_ep_points = np.mean(bot_player_nb_points)
print(f"mean rl_player points : {mean_rl_ep_points}  mean bot_player points : {mean_bot_ep_points}")

x = np.linspace(0, len(rl_player_wins_list), len(rl_player_wins_list))
# Wins evolution figure:
plt.figure(figsize=(15, 5))
plt.title("Number of wins between trained RL and Heuristic Player")
plt.xlabel("Game number")
plt.ylabel("nb wins")
plt.plot(x, rl_player_wins_list, label='rl_player')
plt.plot(x, bot_player_wins_list, label='bot_player')
plt.legend()
plt.show()


