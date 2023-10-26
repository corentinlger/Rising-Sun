from tabulate import tabulate
from typing import Optional, Union, List, Tuple, Any, Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game.players import Player, HeuristicPlayer, SepukuPoetsPlayer, HumanPlayer, bot_player_dict


class GameEnv(gym.Env):
    """
     Custom Environment that follows the gym interface for the board game 'Rising Sun'

     It takes two players from the Players class to initialize a game. The game is played
     in 1v1. Both players have a simultaneous action (in the RL sense of the term) that is
     the number of golds used on each of the game actions (not in the RL sense of the term)
     These game actions are assigned to each player and applied automatically according to
     the 'gold actions' of the players. The four game actions are coded at the end of the
     class.
    """

    def __init__(self,
                 player: Player,
                 bot_player: Player,
                 fights_per_game: Optional[int] = 2,
                 bot_reward_penalty: Optional[float] = 0.5,
                 golds_reward_penalty: Optional[float] = 0.5,
                 verbose: Optional[bool] = False):
        super().__init__()

        self.rl_player = player
        self.rl_player_gold = None
        self.bot_player = bot_player
        self.actions_names = ['Sepuku', 'Hostage', 'Ronins', 'Imperial Poets']
        self.fight_nb = 0
        self.fights_per_game = fights_per_game
        self.death_per_fights = np.zeros(fights_per_game)
        self.bot_reward_penalty = bot_reward_penalty
        self.golds_reward_penalty = golds_reward_penalty
        self.max_gold_per_action = 7

        self.action_space = spaces.Box(low=np.zeros(4),
                                       high=np.ones(4)*self.max_gold_per_action,
                                       dtype=np.int64)

        # We also fix the limits of the values of observation space with the values indicated in the Player class
        max_nb_force_per_fight = 10
        max_golds = 20
        max_ronins = 3
        self.observation_space = spaces.Box(low=np.zeros(9),
                                            high=np.array([fights_per_game,
                                                           max_nb_force_per_fight,
                                                           max_nb_force_per_fight,
                                                           max_nb_force_per_fight,
                                                           max_nb_force_per_fight,
                                                           max_golds,
                                                           max_golds,
                                                           max_ronins,
                                                           max_ronins]),
                                            dtype=np.int64)

        self.verbose = verbose
        if self.verbose: self._show_game_state()

    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:
        # reset les paramètres et ceux des deux joueurs aussi
        self.fight_nb = 0
        self.death_per_fights = np.zeros(self.fights_per_game)
        self.rl_player.reset()
        self.bot_player.reset()
        observation = self._get_observation(self.rl_player)
        self.rl_player_gold = observation[5]

        info = {}
        return observation, info

    def step(self, action: np.array) -> tuple[Any, float | Any, bool, bool, dict[Any, Any]]:
        """
        'action' and 'bot_action' are the golds used by each player on a certain action
        They are called actions because this is what the players do when they play
        We keep this name because of the RL terminology, but it can be confusing

        The actions that actions_assignement refer to are the game actions :
        'sepuku', 'hostage', 'ronins' and 'imperial poets'
        """
        if self.verbose:
            self._show_game_state()

        # Action of the bot player
        bot_obs = self._get_observation(self.bot_player)
        bot_action = self.bot_player.choose_action(bot_obs)

        # Action of the rl player
        action = self._transform_action(action)

        actions_assignement = np.zeros(4)
        for i, gold_balance in enumerate(action - bot_action):
            # rl_player used more golds
            if gold_balance > 0:
                actions_assignement[i] = 1
            elif gold_balance < 0:
                actions_assignement[i] = 2

        reward_rl_player, reward_bot_player = self._apply_actions(assignement_vector=actions_assignement)
        # We want to increase our reward as much as we want to minimize the opponent reward

        reward = self._reward_function(action, reward_rl_player, reward_bot_player)

        observation = self._get_observation(self.rl_player)

        done = self.fight_nb >= self.fights_per_game - 1
        if done:
            # If it is the last fight of the episode, the rl_player get additional reward if he wins the game
            if self.rl_player.nb_points > self.bot_player.nb_points:
                reward += 10

        self.fight_nb += 1

        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _transform_action(self, action):
        """
        Transform the float to int action, and decrease the golds used if they exceed the possessed value
        """
        action = np.rint(action)
        while sum(action) > self.rl_player.golds:
            idx = np.random.randint(len(action))
            action[idx] = max(action[idx]-1, 0)
        return action

    def _reward_function(self, action: np.array, reward_rl: float, reward_bot: float) -> float:
        """
        Calculates the reward for rl_player with its points, the points of his opponent and a penalty
        if he used more golds than available.
        """
        reward = reward_rl
        reward -= reward_bot * self.bot_reward_penalty

        golds_spent = np.sum(action)
        if golds_spent > self.rl_player_gold:
            surplus_gold = golds_spent - self.rl_player_gold
            reward -= surplus_gold * self.golds_reward_penalty

        return reward

    def _get_observation(self, player: Player) -> np.array:
        """
        Returns the observation for a player
        Rt is composed of the fight number an of all the stats
        of the player getting the observation and of its opponent
        """
        opponent_player = self.rl_player if player == self.bot_player else self.bot_player

        return np.array((self.fight_nb,
                         player.force_per_fights[0],
                         player.force_per_fights[1],
                         opponent_player.force_per_fights[0],
                         opponent_player.force_per_fights[1],
                         player.golds,
                         opponent_player.golds,
                         player.nb_ronins,
                         opponent_player.nb_ronins))

    def _apply_actions(self, assignement_vector: np.array) -> tuple[Any, Any]:
        """
        Apply the four possible game actions and update their consequences on players stats.
        First, apply the three first actions, then check which player won the fight, and finally
        apply the last action.
        """
        first_actions = [self._sepuku, self._hostage, self._ronins]
        players = [self.rl_player, self.bot_player]
        for i, player_id in enumerate(assignement_vector[:3]):
            if player_id != 0:
                player_action_i = players[int(player_id - 1)]
                first_actions[i](player_action_i)

        # TODO : add a bonus if already won the last fight
        # Checking of the winner and application of the fight rules
        if self.rl_player.force_per_fights[self.fight_nb] != self.bot_player.force_per_fights[self.fight_nb]:
            if self.rl_player.force_per_fights[self.fight_nb] > self.bot_player.force_per_fights[self.fight_nb]:
                fight_winner, fight_loser = self.rl_player, self.bot_player
            else:
                fight_winner, fight_loser = self.bot_player, self.rl_player
            self.death_per_fights[self.fight_nb] += fight_loser.force_per_fights[self.fight_nb]
            fight_loser.force_per_fights[self.fight_nb] = 0
            fight_loser.golds += fight_winner.gold_used_current_fight
            fight_winner.nb_points += 4

        # Application of the last action
        if assignement_vector[3] != 0:
            player_action_poets = players[int(assignement_vector[3] - 1)]
            self._poets(player_action_poets)

        return self.rl_player.nb_points, self.bot_player.nb_points

    """
    Definition of the 4 available actions for a player :
    """

    def _sepuku(self, player: Player) -> None:
        """
        The units of the player suicide, and he gets 1 point for each unit
        """
        nb_sepuku = player.force_per_fights[self.fight_nb]
        player.nb_points += nb_sepuku
        player.force_per_fights[self.fight_nb] = 0
        self.death_per_fights[self.fight_nb] += nb_sepuku

    def _hostage(self, player: Player) -> None:
        """
        The player captures every enemy units and gets 1 golds per unit captured
        """
        captured_player = self.bot_player if player == self.rl_player else self.rl_player
        captured_units = captured_player.force_per_fights[self.fight_nb]
        captured_player.force_per_fights[self.fight_nb] = 0
        player.golds += captured_units

    def _ronins(self, player: Player) -> None:
        """
        The player increases his number of units by the number of ronins called
        """
        player.force_per_fights[self.fight_nb] += player.nb_ronins

    def _poets(self, player: Player) -> None:
        """
        The player gets 1 point for each unit killed in combat during the fight
        """
        player.nb_points += self.death_per_fights[self.fight_nb]

    def _show_game_state(self):
        rl_name, rl_golds, rl_force_per_fights, rl_nb_ronins, rl_nb_points = self.rl_player.get_statistics()
        bot_name, bot_golds, bot_force_per_fights, bot_nb_ronins, bot_nb_points = self.bot_player.get_statistics()

        rl_force_str = ', '.join(map(str, rl_force_per_fights))
        bot_force_str = ', '.join(map(str, bot_force_per_fights))

        data = [
            ["", rl_name, bot_name],
            ["Golds", rl_golds, bot_golds],
            ["Force per Fights", rl_force_str, bot_force_str],
            ["Number of Ronins", rl_nb_ronins, bot_nb_ronins],
            ["Number of Points", rl_nb_points, bot_nb_points]
        ]

        print("\nCurrent state:")
        print(f'Fight : {self.fight_nb + 1} / {self.fights_per_game}')
        print(tabulate(data, headers="firstrow", tablefmt="fancy_grid"))


def initialize_players(bot_behavior):
    # TODO CHANGE THIS BECAUSE HERE THE PLAYER ISNT A RL PLAYER 


    rl_player = Player(name='rl_player')
    bot_player = bot_player_dict[bot_behavior](name='bot_player')
    return rl_player, bot_player

def initialize_game(args, rl_player, bot_player, verbose=True):
    rl_player.reset()
    bot_player.reset()
    env = GameEnv(player=rl_player, bot_player=bot_player, fights_per_game=args.fights_per_game, verbose=verbose)

    return env



