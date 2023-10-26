import os 

import numpy as np

from utils.algos import algos
from utils.utils import create_saving_directories


class Player:
    """
    This class implement and update the players statistics used by both the RL
    Agent and the Bot player.

    Each player starts the game with random statistics (number of golds, number
    of units per fight, number of ronins) and counters (number of points scored,
    golds used during each fight).

    For the Bot player, this class also implements the actions to return, based
    one hard-coded rules, in order to imitate realistic behaviors to play against
    the Rl Agent.

    The players have a reset method that re-initialize those parameters and that
    is used by the reset method of the gym environment.
    """

    def __init__(self, name, fights_per_game=2):
        self.name = name
        self.fights_per_game = fights_per_game
        self.golds = np.random.randint(5, 10)
        self.gold_used_current_fight = 0
        self.force_per_fights = np.random.randint(1, 6, size=self.fights_per_game)
        self.nb_ronins = np.random.randint(0, 3)
        self.nb_points = 0

    def reset(self):
        self.golds = np.random.randint(5, 10)
        self.gold_used_current_fight = 0
        self.force_per_fights = np.random.randint(1, 6, size=self.fights_per_game)
        self.nb_ronins = np.random.randint(0, 3)
        self.nb_points = 0

    def show_statistics(self):
        print(f"{self.name}   Golds: {self.golds}  Units: {self.force_per_fights}  Ronins: {self.nb_ronins}  Points: {self.nb_points}")

    def get_statistics(self):
        return self.name, self.golds, self.force_per_fights, self.nb_ronins, self.nb_points

    def choose_action(self, state):
        self.gold_used_current_fight = np.random.randint(low=0, high=self.golds)
        golds_per_action = np.zeros(4)
        for i in range(self.gold_used_current_fight):
            action = np.random.randint(0, len(golds_per_action))
            golds_per_action[action] += 1
        self.golds -= self.gold_used_current_fight
        return golds_per_action
    
    
class TrainedPlayer(Player):
    """
    Class to create an opponent with an action policy (behavior) trained with Reinforcement Learning.
    """

    def __init__(self, name, fights_per_game=2, algo="PPO", training_timesteps=100000, seed=0):
        self.name = name
        self.fights_per_game = fights_per_game
        self.golds = np.random.randint(5, 10)
        self.gold_used_current_fight = 0
        self.force_per_fights = np.random.randint(1, 6, size=self.fights_per_game)
        self.nb_ronins = np.random.randint(0, 3)
        self.nb_points = 0
        logs_dir, models_dir = create_saving_directories(fights_per_game)
        agent_name = create_agent_name(algo, training_timesteps, seed)
        self.model = self.load_policy(algo, algos, models_dir, agent_name)

    def load_policy(self, algo, algos, models_dir, agent_name):
        """
        Load a trained RL policy
        :param algo : (str) The algorithm used
        :param models : {(str)} Hashmap of supported stable-baselines3 algorithms
        :param models_dir : (str) Path to the models saving directory
        :param agent_name : (str) Name of the RL agent in models directory
        """
        model_name = os.path.join(models_dir, agent_name)
        try:
            model = algos[algo].load(model_name)
        except:
            raise(ValueError(f"No model has been trained with this configuration yet"))
        return model


    def choose_action(self, state):
        self.gold_used_current_fight = np.random.randint(low=0, high=self.golds)
        golds_per_action = np.zeros(4)
        for i in range(self.gold_used_current_fight):
            action = np.random.randint(0, len(golds_per_action))
            golds_per_action[action] += 1
        self.golds -= self.gold_used_current_fight
        return golds_per_action



class SepukuPoetsPlayer(Player):
    """
    Class to create an opponent with a hard coded behavior.

    At each time step, this player tries to maximize its points by applying the action sepuku (sacrificing its units) and imperial poets (win points per units killed).
    """
    def choose_action(self, state):
        fight_number = state[0]
        if fight_number == 0:
            self.gold_used_current_fight = int(self.golds/2)
        else:
            self.gold_used_current_fight = self.golds
        golds_per_action = np.zeros(4)
        golds_sepuku = int(self.gold_used_current_fight/2)
        golds_poets = self.gold_used_current_fight - golds_sepuku
        golds_per_action[0] += golds_sepuku
        golds_per_action[3] += golds_poets
        self.golds -= self.gold_used_current_fight
        return golds_per_action


class HeuristicPlayer(Player):
    """
    Class to create an opponent with a hard coded behavior.

    If it is the first fight, this player only uses half of his golds. It it is the last one, he uses all golds.
    """
    def choose_action(self, state):
        fight_number = state[0]
        if fight_number == 0:
            # If it is the first fight, use half of the golds
            self.gold_used_current_fight = int(self.golds/2)
        else:
            # Else use all golds available
            self.gold_used_current_fight = self.golds
        golds_per_action = np.zeros(4)
        for i in range(self.gold_used_current_fight):
            action = np.random.randint(0, len(golds_per_action))
            golds_per_action[action] += 1
        self.golds -= self.gold_used_current_fight
        return golds_per_action
    


class HumanPlayer(Player):
    """
    Class to play against a bot player and input your own actions.
    """
    def choose_action(self):
        print("Enter your gold values for sepuku, hostage, ronins and poets separated with spaces : ")
        user_input = input()

        input_list = user_input.split()

        golds_per_action = np.array([int(value) for value in input_list])
        print(f"{golds_per_action = }")
        
        return golds_per_action

# TODO : Problem with that because I shouldn't pass the seed in the initialization .... See the code used in ER-MRL 
def create_agent_name(algo, training_timesteps):
    return f"{algo}_{int(training_timesteps/1000)}k_steps"


bot_player_dict = {"random" : Player,
                   "heuristic": HeuristicPlayer,
                   "sepuku_poets": SepukuPoetsPlayer}

