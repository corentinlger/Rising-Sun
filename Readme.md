# Rising Sun project

- To try to beat friends on the combat phases of the "Rising Sun" board game, I created this simplified gymnasium implementation of the game.
The project also incorporates a python script to train a single RL agent on the environment. 
This agent can currently only be trained against hard-coded players (either random of with heuristics).

- You can train an agent with the `train.py` file, and then test its performance with `evaluate_model.py`. 
If you wish to code your own agent with a scripted behavior, you can do it by creating a new agent class inheriting from the Player class in `players.py`.

- You can also play against an agent with a trained or scrpited action policy with `play_vs_bot.py`.

- The next step will be to add multi-agents training to the project.
Then, it will be interesting to explore the path of self play to see how complex the behaviors of the agent can become !