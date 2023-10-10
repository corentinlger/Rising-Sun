import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from game_env import GameEnv
from players import Player, HeuristicPlayer

# Trained model loading
model_path = 'models/seed_0_100k_steps.zip'
model = PPO.load(model_path)

# Players and Env creation
rl_player = Player('rl_player')
bot_player = HeuristicPlayer('bot_player')
env = GameEnv(rl_player, bot_player, verbose=True)

# Testing parameters
nb_testing_games = 1000

# Creation of lists to score the victories
rl_player_wins, rl_player_wins_list, rl_player_nb_points = 0, [], []
bot_player_wins, bot_player_wins_list, bot_player_nb_points = 0, [], []

for game_nb in range(nb_testing_games):
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


