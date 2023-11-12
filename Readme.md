# Rising Sun project

To try to beat friends on the combat phases of the ["Rising Sun"](https://en.wikipedia.org/wiki/Rising_Sun_(board_game)) board game, I created this simplified gymnasium implementation of the game.
The project also incorporates python scripts to train and evaluate RL agents against bots. 
You can also play against bots with trained or scripted behaviors on a web application.

## Installation 

1- Get the repository

```bash
git clone git@github.com:corentinlger/Rising-Sun.git
cd Rising-Sun
```

2- Install the dependencies

```bash
python -m venv myvenv
myvenv\Scripts\activate.bat
pip install -r requirements.txt
```


## Usage

Train an agent against a bot

```bash
python train.py --algo PPO --bot_behavior random --total_timesteps 100000 --nb_seeds 3
```

Evaluate a trained agent against a bot

```bash
python evaluate_model.py --algo PPO --bot_behavior random --tr_bot_behavior random --tr_timesteps 100000 --seed 0
```

Play against trained or scripted agents on a web application 

```bash
python app.py
```

If you wish to code your own agent with a scripted behavior, you can do it by creating a new agent class inheriting from the Player class in `game/players.py`.

## Development

- The next step will be to add multi-agents training to the project.
Then, it will be interesting to explore the path of self play to see how complex the behaviors of the agent can become !