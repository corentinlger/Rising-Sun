import os
import argparse

from stable_baselines3 import PPO

from game.game_env import GameEnv
from game.players import HumanPlayer, bot_player_dict


def get_player_name():
    """
    Welcome a player and get his name 
    """
    print("Hello and welcome to this version of the fighting phase of Rising Sun board game !")
    player_name = input("\nEnter your name : ")
    return player_name

def initialize_players(player_name, bot_behavior):
    """
    Initialize players
    :param player_name: (str) the name of the human player
    :param bot_behavior: (str) the behavior of the bot
    :return player: (Player) instance of the human player
    :return bot_player: (Player) instance of the bot player
    """
    player = HumanPlayer(name=player_name)
    
    if bot_behavior in bot_player_dict:
        bot_player = bot_player_dict[bot_behavior](name='bot_player')
    elif bot_behavior == "trained":
        try:
            pass
        except:
            raise(ValueError("A trained agent with those parameters hasn't been found"))
    else:
        raise(ValueError("Unknown bot bahavior"))

    return player, bot_player

def ask_displaying_rules():
    """
    Ask the player if he wants the rules to be displayed
    """
    print("\nDo you want to read the rules (Y/N) ?")    
    display_rules = None

    while display_rules not in ["Y", "N"]:
        display_rules = input()

    return True if display_rules == "Y" else False
     
def display_rules():
    """
    Display the rules if the player asked
    """
    with open('utils/rules.txt', 'r') as file:
            print("\n ---- RULES ----")
            print(file.read())
            input("Press any key to continue...")

def play_game(player, bot_player, fights_per_game, nb_games):
    """
    Play against a bot player and return the number of victories at the end 
    :param player: (Player) instance of the human player
    :param bot_player: (Player) instance of the bot player
    :param fights_per_game: (int) Number of fights in a game
    :param nb_games: (int) Number of games played vs the bot
    :return player_won_games: (int) The number of games won by the human player 
    :return bot_won_games: (int) The number of games won by the bot player
    """
    print(f"\nBeginning of the game")
    player_won_games = bot_won_games = 0
    env = GameEnv(player=player, bot_player=bot_player, fights_per_game=fights_per_game, verbose=True)

    for _ in range(nb_games):
        print("")
        obs, info = env.reset()
        done = False
        player_ep_nb_points = []
        bot_ep_nb_points = []
        while not done:
            action =  player.choose_action()
            obs, reward, done, truncated, info = env.step(action)
            player_ep_nb_points.append(player.nb_points)
            bot_ep_nb_points.append(bot_player.nb_points)
            print(f"\nplayer points : {player.nb_points} bot_player points : {bot_player.nb_points}")
        if player.nb_points > bot_player.nb_points:
            player_won_games += 1
        elif player.nb_points < bot_player.nb_points:
            bot_won_games += 1

    return player_won_games, bot_won_games

def print_game_result(player_won_games, bot_won_games):
    """
    Print the results of the human vs bot confrontation
    :param player_won_games: (int) The number of games won by the human player 
    :param bot_won_games: (int) The number of games won by the bot player
    """
    print(f"\nFinal result : ")
    if player_won_games == bot_won_games:
        print(f"Equailty : you both won {player_won_games} fights")
    elif player_won_games > bot_won_games:
        print(f"Victory ! You won {player_won_games} fights and your opponent won {bot_won_games} fights")
    elif player_won_games < bot_won_games:
        print(f"Defeat ! You won {player_won_games} fights and your opponent won {bot_won_games} fights")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--nb_games", type=int, required=False, default=3)
    parser.add_argument("--fights_per_game", type=int, required=False, default=2)
    parser.add_argument("--bot_behavior", type=str, required=False, default="random")
    parser.add_argument("--training_timesteps", type=int, required=False, default=100000)
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--algo", type=str, required=False, default="PPO")

    args = parser.parse_args()

    player_name = get_player_name()

    player, bot_player = initialize_players(player_name, args.bot_behavior)

    if ask_displaying_rules():
         display_rules()
        
    player_won_games, bot_won_games = play_game(player, bot_player, args)

    print_game_result(player_won_games, bot_won_games)


