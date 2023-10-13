from flask import Flask, render_template, request, jsonify

from game.game_env import GameEnv
from game.players import Player, HeuristicPlayer, SepukuPoetsPlayer, HumanPlayer
from play_vs_bot import welcome_player, initialize_players, ask_displaying_rules, display_rules, play_game, print_game_result


app = Flask(__name__)

# Existing game logic goes here

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play', methods=['POST'])
def play():
    # Get user input from request.form and update game state
    player, bot_player = initialize_players(request.form['bot_behavior'])
    player_won_fights, bot_won_fights = play_game(player, bot_player, request.form)
    result = {'player_won_fights': player_won_fights, 'bot_won_fights': bot_won_fights}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)