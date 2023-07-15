import numpy as np


class Player:


    """
    This class implement and update the players statistics used by both the RL
    Agent and the Bot player

    Each player starts the game with random statistics (number of golds, number
    of units per fight, number of ronins) and counters (number of points scored,
    golds used during each fight).

    For the Bot player, this class also implements the actions to return, based
    one hard-coded rules, in order to imitate realistic behaviors to play against
    the Rl Agent

    The players have a reset method that re-initialize those parameters and that
    is used by the reset method of the gym environment
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
        # print(f'{self.name} : {self.gold_used_current_fight} golds used on {self.golds}')
        golds_per_action = np.zeros(4)
        for i in range(self.gold_used_current_fight):
            action = np.random.randint(0, len(golds_per_action))
            golds_per_action[action] += 1
        self.golds -= self.gold_used_current_fight
        return golds_per_action


class SepukuPoetsPlayer(Player):

    def choose_action(self, state):
        if state[0] == 0:
            self.gold_used_current_fight = self.golds // 2
        else:
            self.gold_used_current_fight = self.golds
        # print(f'{self.name} : {self.gold_used_current_fight} golds used on {self.golds}')
        golds_per_action = np.zeros(4)
        golds_sepuku = self.gold_used_current_fight // 2
        golds_poets = self.gold_used_current_fight - golds_sepuku
        golds_per_action[0] += golds_sepuku
        golds_per_action[3] += golds_poets
        self.golds -= self.gold_used_current_fight
        return golds_per_action


class HeuristicPlayer(Player):

    def choose_action(self, state):
        # state[0] = numéro du combat
        if state[0] == 0:
            self.gold_used_current_fight = self.golds // 2  # si c'est le premier on utilise la moitié de nos golds
        else:
            self.gold_used_current_fight = self.golds  # si c'est le dernier on utilise tous nos golds
        # print(f'{self.name} : {self.gold_used_current_fight} golds used on {self.golds}')
        golds_per_action = np.zeros(4)
        for i in range(self.gold_used_current_fight):
            action = np.random.randint(0, len(golds_per_action))
            golds_per_action[action] += 1
        self.golds -= self.gold_used_current_fight
        return golds_per_action

