from flask import Flask, render_template, request, jsonify

from game.game_env import GameEnv
from game.players import Player, HeuristicPlayer, SepukuPoetsPlayer, HumanPlayer
from play_vs_bot import initialize_players, ask_displaying_rules, display_rules, play_game, print_game_result


app = Flask(__name__)
game_initialized = False

# Existing game logic goes here

@app.route('/')
def index():

    global game_initialized
    game_initialized = False

    return render_template('index.html')

@app.route('/rules')
def rules():
    return render_template('rules.html')

@app.route('/play', methods=['POST'])
def play():

    global game_initialized

    player_name = request.form['player_name']
    bot_behavior = request.form['bot_behavior']
    fights_per_game = int(request.form['fights_per_game'])
    nb_games = (request.form['nb_games'])

    if not game_initialized:
        player, bot_player = initialize_players(player_name, bot_behavior)
        env = GameEnv(player=player, bot_player=bot_player, fights_per_game=fights_per_game, verbose=False)
        game_initialized = True
    else:
        player, bot_player = env.player, env.bot_player

    player_state = player.get_statistics()
    bot_state = bot_player.get_statistics()

    # state = name, golds, forces_per_fight, nb_ronins, nb_points

    # TODO : Display players states in a clean table 

    # TODO : Enable the player to take an action by filling a kinf of form an d submitting it 

    # TODO : Calcultate the next state of the game and return it 

    # TODO : When the number of games is finished, print a message to show who won and propose to make a rematch

    # TODO : Add a message to ask if the player is sure to wanna stop the game (surely with javascript idk lets go)

    return render_template('play.html', player_state=player_state, bot_state=bot_state)

if __name__ == '__main__':
    app.run(debug=True)