import numpy as np
from flask import Flask, render_template, request, redirect

from game.game_env import GameEnv
from play_vs_bot import initialize_players

def initialize_game(player_name, bot_behavior, fights_per_game=2):
    """
    Initialize two players and an instance of the game environment
    :param player_name: (str) name of the player
    :param bot_behavior: (str) the bot_behavior (i.e. its action policy)
    :param fights_per_game: (int) the number of fights during a game
    :return env: (gym.Env) the environment integrating both players
    """
    global env, player, bot_player
    player, bot_player = initialize_players(player_name, bot_behavior)
    env = GameEnv(player=player, bot_player=bot_player, fights_per_game=fights_per_game, verbose=False)
    env.reset()
    return env

def get_game_status(player, bot_player):
    """
    Return the status at the end of a game (victory, defeat or equality)
    :param player: (str) the human player
    :param bot_player: (str) the bot player
    :return game_status: (str) the game status 
    """
    if player.nb_points > bot_player.nb_points:
        return "Victory"
    elif player.nb_points < bot_player.nb_points:
        return "Defeat"
    else:
        return "Equality"

app = Flask(__name__)

@app.route('/')
def index():
    global game_initialized, player_name, bot_behavior, fights_per_game
    game_initialized = False
    player_name = None
    bot_behavior = None
    fights_per_game = None
    return render_template('index.html')

@app.route('/rules')
def rules():
    global game_initialized
    game_initialized = False
    return render_template('rules.html')

@app.route('/end_game')
def end_game():
    global game_initialized
    game_initialized = False

    game_status = request.args.get('status')
    return render_template('end_game.html', status=game_status)


@app.route('/play', methods=['GET', 'POST'])
def play():
    global game_initialized, env, player, bot_player, player_name, bot_behavior, fights_per_game

    # If the game hasn't been initialized, create a new one
    if not game_initialized:
        # And that game parameters aren't initialized either (i.e it isn't a rematch where the player already specified these parameters)
        if request.method == 'POST' and (not player_name or not bot_behavior or not fights_per_game):
            player_name = request.form['player_name']
            bot_behavior = request.form['bot_behavior']
            fights_per_game = int(request.form['fights_per_game'])

        env = initialize_game(player_name, bot_behavior, fights_per_game)

    # If the game is already initialized, play until the end of the game
    if request.method =='POST' and game_initialized == True:
        action = np.array([int(request.form['Sepuku']),
                        int(request.form['Hostage']),
                        int(request.form['Ronins']),
                        int(request.form['Imperial_Poets'])])
        obs, reward, done, truncated, info = env.step(action)

        # return the game status at the end 
        if done or truncated:
            game_status = get_game_status(player, bot_player)
            return redirect(f'/end_game?status={game_status}')
        
    # Get the players statistics to display them on the web interface
    game_initialized = True 
    player_state = player.get_statistics()
    bot_state = bot_player.get_statistics()
    
    return render_template('play.html', player_state=player_state, bot_state=bot_state, nb_fights=env.fights_per_game, fight_nb=env.fight_nb+1)

if __name__ == '__main__':
    app.run(debug=True)


 # TODO : Allow playing against trained RL agents and not only bots with scripted behaviors 
