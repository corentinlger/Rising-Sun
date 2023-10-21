import numpy as np

from flask import Flask, render_template, request, jsonify

from game.game_env import GameEnv
from play_vs_bot import initialize_players


app = Flask(__name__)
# game_initialized = False

@app.route('/')
def index():
    # global game_initialized
    # game_initialized = False
    return render_template('index.html')

@app.route('/rules')
def rules():
    return render_template('rules.html')

@app.route('/play', methods=['POST'])
def play():

    # global game_initialized
    game_initialized = False

    done = False
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

    while not done:

        player_state = player.get_statistics()
        bot_state = bot_player.get_statistics()

        action = np.array(int(request.form['Sepuku']),
                        int(request.form['Hostage']),
                        int(request.form['Ronins']),
                        int(request.form['Imperial_Poets']))
        

        obs, reward, done, truncated, info = env.step(action)


    # TODO : Enable the player to take an action by filling a kinf of form an d submitting it 

    # TODO : Calcultate the next state of the game and return it 

    # TODO : When the number of games is finished, print a message to show who won and propose to make a rematch

    # TODO : Add a message to ask if the player is sure to wanna stop the game (surely with javascript idk lets go)

    return render_template('game_state.html', player_state=player_state, bot_state=bot_state)

if __name__ == '__main__':
    app.run(debug=True)