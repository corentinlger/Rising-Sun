import numpy as np


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
        print(f"Bot action : {golds_per_action} nb_gold_action : {np.sum(golds_per_action)}/{self.golds}")
        self.golds -= self.gold_used_current_fight
        return golds_per_action

