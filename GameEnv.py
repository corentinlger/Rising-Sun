import gym
import numpy as np
from typing import Optional, Union, List, Tuple, Any, Dict
from gym import spaces
from gym.core import RenderFrame
from Players import Player


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
                 rl_agent_player: Player,
                 bot_player: Player,
                 fights_per_game: Optional[int] = 2,
                 bot_reward_penalty: Optional[float] = 0.5,
                 golds_reward_penalty: Optional[float] = 0.5,
                 verbose: Optional[bool] = False):
        super().__init__()

        self.rl_player = rl_agent_player
        self.bot_player = bot_player
        self.actions_names = ['Sepuku', 'Hostage', 'Ronins', 'Imperial Poets']
        self.fight_nb = 0
        self.fights_per_game = fights_per_game
        self.death_per_fights = np.zeros(fights_per_game)
        self.bot_reward_penalty = bot_reward_penalty
        self.golds_reward_penalty = golds_reward_penalty
        self.max_gold_per_action = 7

        self.action_space = spaces.MultiDiscrete([self.max_gold_per_action] * 4, dtype=int)

        # We also fix the limits of the values of observation space with the values indicated in the Player class

        max_nb_force_per_fight = 10
        max_golds = 20
        max_ronins = 3
        self.observation_space = spaces.MultiDiscrete([fights_per_game,
                                                       max_nb_force_per_fight,
                                                       max_nb_force_per_fight,
                                                       max_nb_force_per_fight,
                                                       max_nb_force_per_fight,
                                                       max_golds,
                                                       max_golds,
                                                       max_ronins,
                                                       max_ronins], dtype=int)

        self.verbose = verbose
        if self.verbose: self.show_game_state()

    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:
        # reset les paramètres et ceux des deux joueurs aussi
        self.fight_nb = 0
        self.death_per_fights = np.zeros(self.fights_per_game)
        self.rl_player.reset()
        self.bot_player.reset()
        observation = self.get_observation(self.rl_player)
        self.rl_player_gold = observation[5]

        info = {}
        return observation, info

    def step(self, action: np.array) -> tuple[Any, float | Any, bool, bool, dict[Any, Any]]:
        """
        'action' and 'bot_action' are the golds used by each player on a certain action
        They are called actions because this is what the players do when they play
        We keep this name because of the RL terminology, but it can be consufing

        The actions that actions_assignement refer to are the game actions :
        'sepuku', 'hostage', 'ronins' and 'imperial poets'
        """

        # Action of the bot player
        bot_obs = self.get_observation(self.bot_player)
        bot_action = self.bot_player.choose_action(bot_obs)

        # Golds used checking
        available_gold = self.rl_player_gold
        golds_spent = np.sum(action)
        if golds_spent > available_gold:
            # Rescale the action and transform it back into a int vector
            action = (golds_spent / available_gold) * action
            action = np.rint(action)

        actions_assignement = np.zeros(4)
        for i, gold_balance in enumerate(action - bot_action):
            # rl_player used more golds
            if gold_balance > 0:

                actions_assignement[i] = 1
            elif gold_balance < 0:
                actions_assignement[i] = 2

        reward_rl_player, reward_bot_player = self.apply_actions(assignement_vector=actions_assignement)
        # We want to increase our reward as much as we want to minimize the opponent reward
        reward = self.reward_function(action, reward_rl_player, reward_bot_player)

        # We get the new observation and update the rl_player_gold
        observation = self.get_observation(self.rl_player)
        self.rl_player_gold = observation[5]

        done = self.fight_nb >= self.fights_per_game - 1
        if done:
            # If it is the last fight of the episode, the rl_player get additional reward if he wins
            if self.rl_player.nb_points > self.bot_player.nb_points:
                reward += 10

        self.fight_nb += 1

        info = {}
        truncated = False

        if self.verbose: self.show_game_state()

        return observation, reward, done, truncated, info

    def reward_function(self, action: np.array, reward_rl: float, reward_bot: float) -> float:
        # reward on points scored and enemy points scored
        reward = reward_rl
        reward -= reward_bot * self.bot_reward_penalty
        # reward on nb golds used (=negative if used more golds than possesed)
        golds_spent = np.sum(action)
        if golds_spent > self.rl_player_gold:
            surplus_gold = golds_spent - self.rl_player_gold
            reward -= surplus_gold * self.golds_reward_penalty

        return reward

    def get_observation(self, player: Player) -> np.array:
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

    def apply_actions(self, assignement_vector: np.array) -> tuple[Any, Any]:

        # Application of the 3 first actions : sepuku, hostage and ronins
        first_actions = [self.sepuku, self.hostage, self.ronins]
        players = [self.rl_player, self.bot_player]
        for i, player_id in enumerate(assignement_vector[:3]):
            if player_id != 0:
                player_action_i = players[int(player_id - 1)]
                first_actions[i](player_action_i)

        # Checking of the winner and application of the fight rules
        if self.rl_player.nb_points != self.bot_player.nb_points:
            if self.rl_player.nb_points > self.bot_player.nb_points:
                fight_winner, fight_loser = self.rl_player, self.bot_player
            else:
                fight_winner, fight_loser = self.bot_player, self.rl_player

            self.death_per_fights[self.fight_nb] += fight_loser.force_per_fights[self.fight_nb]
            fight_loser.golds += fight_winner.gold_used_current_fight
            fight_winner.nb_points += 4

        # Application of the last action
        if assignement_vector[3] != 0:
            self.poets(players[int(assignement_vector[3] - 1)])

        return self.rl_player.nb_points, self.bot_player.nb_points

    """
    Definition of the 4 available actions for a player :
    """

    def sepuku(self, player: Player) -> None:
        """
        The units of the player suicide, and he gets 1 point for each unit
        """
        nb_sepuku = player.force_per_fights[self.fight_nb]
        player.nb_points += nb_sepuku
        player.force_per_fights[self.fight_nb] = 0
        self.death_per_fights[self.fight_nb] += nb_sepuku

    def hostage(self, player: Player) -> None:
        """
        The player captures every enemy units and gets 1 golds per unit captured
        """
        captured_player = self.bot_player if player == self.rl_player else self.rl_player
        captured_units = captured_player.force_per_fights[self.fight_nb]
        captured_player.force_per_fights[self.fight_nb] = 0
        player.golds += captured_units

    def ronins(self, player: Player) -> None:
        """
        The player increases his number of units by the number of ronins called
        """
        player.force_per_fights[self.fight_nb] += player.nb_ronins

    def poets(self, player: Player) -> None:
        """
        The player gets 1 point for each unit killed in combat during the fight
        """
        player.nb_points += self.death_per_fights[self.fight_nb]

    def show_game_state(self):
        print('Game state : ')
        print(f'Fight : {self.fight_nb + 1} / {self.fights_per_game}   Deaths during fights : {self.death_per_fights}')
        self.rl_player.show_statistics()
        self.bot_player.show_statistics()

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass
