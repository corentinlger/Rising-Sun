import numpy as np
from flask import Flask, render_template, request, redirect

from game.game_env import GameEnv
from play_vs_bot import initialize_players

def initialize_game(player_name, bot_behavior, fights_per_game=2):
    global env, player, bot_player
    player, bot_player = initialize_players(player_name, bot_behavior)
    env = GameEnv(player=player, bot_player=bot_player, fights_per_game=fights_per_game, verbose=False)
    env.reset()

app = Flask(__name__)

@app.route('/')
def index():
    global game_initialized
    game_initialized = False
    return render_template('index.html')

@app.route('/rules')
def rules():
    return render_template('rules.html')

@app.route('/end_game')
def end_game():
    return render_template('end_game.html')


@app.route('/play', methods=['GET', 'POST'])
def play():
    global game_initialized, env, player, bot_player

    if request.method == 'POST' and not game_initialized:
        player_name = request.form['player_name']
        bot_behavior = request.form['bot_behavior']
        fights_per_game = int(request.form['fights_per_game'])
        nb_games = (request.form['nb_games'])

        initialize_game(player_name, bot_behavior, fights_per_game)

    if request.method =='POST' and game_initialized == True:
        action = np.array([int(request.form['Sepuku']),
                        int(request.form['Hostage']),
                        int(request.form['Ronins']),
                        int(request.form['Imperial_Poets'])])
        obs, reward, done, truncated, info = env.step(action)

        if done or truncated:
            return redirect('/end_game')
        
    game_initialized = True
    player_state = player.get_statistics()
    bot_state = bot_player.get_statistics()
     
    return render_template('play.html', player_state=player_state, bot_state=bot_state)


if __name__ == '__main__':
    app.run(debug=True)


 # TODO : Enable the player to take an action by filling a kinf of form an d submitting it 

# TODO : Calcultate the next state of the game and return it 

# TODO : When the number of games is finished, print a message to show who won and propose to make a rematch

# TODO : Add a message to ask if the player is sure to wanna stop the game (surely with javascript idk lets go)
