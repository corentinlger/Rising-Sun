import os
import argparse

from stable_baselines3 import PPO

from game_env import GameEnv
from players import Player, HeuristicPlayer, SepukuPoetsPlayer, HumanPlayer

def welcome_player():
    print("Hello and welcome to this version of the fighting phase of Rising Sun board game !")

def initialize_players(args):
    player_name = input("\nEnter your name : ")
    player = HumanPlayer(name=player_name)

    bot_player_dict = {"random" : Player,
                       "heuristic": HeuristicPlayer,
                       "sepuku_poets": SepukuPoetsPlayer}
    
    if args.bot_behavior in bot_player_dict:
        bot_player = bot_player_dict[args.bot_behavior](name='bot_player')
    else:
        raise(ValueError("Unknown bot bahavior, RL opponents are not implemented yet"))

    return player, bot_player


def ask_displaying_rules():
    print("\nDo you want to read the rules (Y/N) ?")    
    display_rules = None

    while display_rules not in ["Y", "N"]:
        condition = display_rules != "Y" or "N"
        display_rules = input()

    return True if display_rules == "Y" else False
     
def display_rules():
    with open('rules.txt', 'r') as file:
            print("\n ---- RULES ----")
            print(file.read())
            input("Press any key to continue...")

def play_game(player, bot_player, args):
    print(f"\nBeginning of the game")
    # Initialize the environment 
    player_won_fights = bot_won_fights = 0
    env = GameEnv(rl_agent_player=player, bot_player=bot_player, fights_per_game=args.fights_per_game, verbose=True)

    for fight in range(args.nb_fights):
        print("")
        obs, info = env.reset()
        done = False
        player_ep_nb_points = []
        bot_ep_nb_points = []
        while not done:
            print("")
            action =  player.choose_action()
            obs, reward, done, truncated, info = env.step(action)
            player_ep_nb_points.append(player.nb_points)
            bot_ep_nb_points.append(bot_player.nb_points)
            print(f"player points : {player.nb_points} bot_player points : {bot_player.nb_points}")
        if player.nb_points > bot_player.nb_points:
            player_won_fights += 1
        elif player.nb_points < bot_player.nb_points:
            bot_won_fights += 1

    return player_won_fights, bot_won_fights

def print_game_result(player_won_fights, bot_won_fights):
    print(f"\nFinal result : ")
    if player_won_fights == bot_won_fights:
        print(f"Equailty : you both won {player_won_fights} fights")
    elif player_won_fights > bot_won_fights:
        print(f"Victory ! You won {player_won_fights} fights and your opponent won {bot_won_fights} fights")
    elif player_won_fights < bot_won_fights:
        print(f"Defeat ! You won {player_won_fights} fights and your opponent won {bot_won_fights} fights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nb_fights", type=int, required=False, default=3)
    parser.add_argument("--fights_per_game", type=int, required=False, default=2)
    parser.add_argument("--bot_behavior", type=str, required=False, default="random")
    parser.add_argument("--training_timesteps", type=int, required=False, default=100000)
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--algo", type=str, required=False, default="PPO")

    args = parser.parse_args()

    welcome_player()

    player, bot_player = initialize_players(args.bot_behavior)

    if ask_displaying_rules():
         display_rules()
        
    player_won_fights, bot_won_fights = play_game(player, bot_player, args)

    print_game_result(player_won_fights, bot_won_fights)


