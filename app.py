from flask import Flask, render_template, request, jsonify

from game.game_env import GameEnv
from game.players import Player, HeuristicPlayer, SepukuPoetsPlayer, HumanPlayer
from play_vs_bot import initialize_players, ask_displaying_rules, display_rules, play_game, print_game_result


app = Flask(__name__)

# Existing game logic goes here

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play', methods=['POST'])
def play():
    player_name = request.form['player_name']
    bot_behavior = request.form['bot_behavior']

    # TODO : When the player clicks the 'play' button, start the game (potentially use a new page / modify the aspect of current page)

    # Initialize the two players
    player, bot_player = initialize_players(player_name, bot_behavior)

    # TODO : Use a form or adapted component to ask if the player wants the rules to be displayed, if yes display them with text on the game 

    # TODO : When previous step if finished, start the game 

    # TODO : This means at each timestep the statistics (=state) of the game are displayed, and player needs to take an action 
    # TODO : To take an action, input 4 numbers (=golds per action) and retrieve those numbers to calculate next state and rewards
    # TODO : When action is processed, return new game state to the user and let him choose his next action 
    # TODO : When the number of games is finished, print a message to show who won and propose to make a rematch
    return render_template('play.html')

if __name__ == '__main__':
    app.run(debug=True)