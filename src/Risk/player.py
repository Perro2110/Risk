"""
players.py — Player classes for the Risk game engine.

Each player subclass implements a distinct strategy via `choose_action`.
The base `Player` class handles turn flow (place → attack → fortify phases).
"""

from abc import ABC, abstractmethod
from Risk.actions import Action, PlaceArmyAction, AttackAction, FortifyAction
from Risk.game_state import GameState
from Risk import utils
from matplotlib import pyplot as plt


###############################################################################
# TODO: implement different player classes with different strategies,         #
# for now we will implement only dummy players that                           #
#                                               always end their turn without #
#                                                                             #
###############################################################################
# class TreePlayer(Player):                                                   #
#     """ Trees player class """                                              #
#     def choose_action(self, game_state: GameState) -> Action|None:          #
#         """                                                                 #
#             This is a dummy implementation of the choose_action method, it  #
#             always returns None (end turn) but in the future it will be     #
#             implemented with a more sophisticated strategy.                 #
#         """                                                                 #
#         return evaluate(game_state)                                         #
###############################################################################
class Player(ABC):
    """
    Abstract base class for all Risk players.

    Manages turn state (current phase, troops remaining) and drives the
    game loop via `play_turn`. Concrete subclasses must implement
    `choose_action` to define their strategy.

    Attributes:
        color (str): The player's color identifier.
        troops_to_place (int): Armies remaining to place in the current turn.
        completed_phases (list[int]): Phases already finished this turn.
    """

    def __init__(self, color: str, troops_to_place: int = 0):
        self.color = color
        self.troops_to_place = troops_to_place
        self.completed_phases: list[int] = []

    @abstractmethod
    def choose_action(self, game_state: GameState) -> Action | None:
        """
        Choose the next action to execute, or None to end the current phase.

        Called repeatedly by `play_turn` until None is returned, at which
        point the game advances to the next phase.
        """
        pass

    def add_completed_phase(self, phase: int):
        """Mark a phase as completed so it won't be re-entered this turn."""
        self.completed_phases.append(phase)

    def set_completed_phase(self, phases: list = []):
        """Mark a phase as completed so it won't be re-entered this turn."""
        self.completed_phases = phases

    def set_troops_to_place(self, troops_to_place: int):
        """Override the number of troops available to place."""
        self.troops_to_place = troops_to_place

    def is_phase_applicable(self, game_state: GameState, phase: int) -> bool:
        """
        Return True if the game is in `phase` and it hasn't been completed yet.
        """
        return game_state.get_phase() == phase and \
            phase not in self.completed_phases

    def __str__(self) -> str:
        return self.__class__.__name__

    ###########################################################################
    #                                                                         #
    #                         !!! NOTICE !!!                                  #    
    #                                                                         #            
    ###########################################################################
    # TODO: evaluate where we want put this code here or in the Game class,   #
    # maybe we want to put it in the Game class and call it from the Player   #
    # class but for now we put it here for simplicity                         #
    ###########################################################################
    def play_turn(self, game_state: GameState, visualizer=None):
        """
        Drive a full turn: place armies → attack → fortify.

        Calls `choose_action` in a loop. Returning None advances the phase;
        the turn ends when phases wrap back to 0.
        """
        game_state.set_phase(0)
        self.troops_to_place = game_state.get_game_map().get_reward(self.color)
        while True:
            action = self.choose_action(game_state)

            if action is not None:
                action.execute()
                if visualizer is not None:
                    visualizer.show(False)
                    visualizer.update(game_state)
                    plt.pause(0.1)
            else:
                game_state.next_phase()
                if game_state.get_phase() == 0:
                    return


class RedPlayer(Player):
    """Red player — Comunist strategy (stub, always passes)."""

    def choose_action(self, game_state: GameState) -> Action | None:
        return None


class PurplePlayer(Player):
    """Purple player — Pixie strategy (stub, always passes)."""

    def choose_action(self, game_state: GameState) -> Action | None:
        return None


class YellowPlayer(Player):
    """Yellow player — Cluster strategy (stub, always passes)."""

    def choose_action(self, game_state: GameState) -> Action | None:
        return None


class GreenPlayer(Player):
    """Green player — Stinky strategy (stub, always passes)."""

    def choose_action(self, game_state: GameState) -> Action | None:
        return None


class BluePlayer(Player):
    """Blue player — Neferius strategy (stub, always passes)."""

    def choose_action(self, game_state: GameState) -> Action | None:
        return None


class BlackPlayer(Player):
    """
    Black player — Angry strategy.

    An aggressive bot that maximises pressure on enemy territories:

    - **Place**: Drops all armies on the owned country with the most
      enemy neighbors.
    - **Attack**: Each owned country attacks its weakest enemy neighbor
      whenever it has numerical superiority and more than 1 army.
      After capturing a territory, armies are shifted toward the new
      frontline if it is more exposed than the source country.
    - **Fortify**: Pulls armies from lower-threat countries toward
      higher-threat friendly neighbors to reinforce the front.
    """

    def __handle_placearmy_phase(self, game_state: GameState) -> Action | None:
        """Place all available troops on the most contested owned country."""
        troops = self.troops_to_place
        if troops == 0:
            return None

        country_to_place = utils.get_most_contested_country(
            game_state,
            self.color
        )
        if country_to_place is None:
            return None

        self.troops_to_place -= troops
        self.add_completed_phase(GameState.PLACE_ARMY)
        return PlaceArmyAction(country_to_place, troops)

    def __handle_attack_phase(self, game_state: GameState) -> Action | None:
        """
        Attack with every eligible country.

        A country is eligible if it has a weaker enemy neighbor and holds
        more than 1 army. Up to 3 armies attack; post-attack movement
        favors the newly captured country if it is more exposed.
        """
        owned_countries = game_state.get_game_map().get_owned_countries(
            self.color
        )

        for country in owned_countries:
            weakest_en = utils.weakest_enemy_neighbour(country)

            if weakest_en is not None and \
                    country.get_army_size() > weakest_en.get_army_size() and \
                    country.get_army_size() > 1:

                num_armies_to_attack = min(3, country.get_army_size() - 1)

                # Default: keep armies at the source after the attack
                num_armies_want_to_move_post_attack = (
                    country.get_army_size() - 1 - num_armies_to_attack
                )

                # Move armies forward if the captured territory is more exposed
                if country.get_number_of_enemy_neighbors() - 1 > \
                        weakest_en.get_number_of_enemy_neighbors(self.color):
                    num_armies_want_to_move_post_attack = 0

                return AttackAction(
                    country,
                    weakest_en,
                    num_armies_to_attack,
                    num_armies_want_to_move_post_attack
                )

        return None  # No valid attacks found; end attack phase

    def __handle_fortify_phase(self, game_state: GameState) -> Action | None:
        """
        Reinforce the front by moving armies from safer to more threatened
        connected friendly countries.
        """
        owned_countries = game_state.get_game_map().get_owned_countries(
            self.color
        )

        # Prioritise countries with the most enemy neighbors as destinations
        sorted_countries = sorted(
            owned_countries,
            key=lambda c: c.get_number_of_enemy_neighbors(),
            reverse=True
        )

        for to_country in sorted_countries:
            if not to_country.get_neighbors():
                continue

            for from_country in to_country.get_connected_friendly_countries():
                if from_country.get_army_size() <= 1:
                    continue

                num_armies_to_move = from_country.get_army_size() - 1
                self.add_completed_phase(GameState.FORTIFY)
                return FortifyAction(from_country,
                                     to_country,
                                     num_armies_to_move
                                     )

    def choose_action(self, game_state: GameState) -> Action | None:
        """Dispatch to the appropriate phase handler."""
        if self.is_phase_applicable(game_state, GameState.PLACE_ARMY):
            return self.__handle_placearmy_phase(game_state)
        elif self.is_phase_applicable(game_state, GameState.ATTACK):
            return self.__handle_attack_phase(game_state)
        elif self.is_phase_applicable(game_state, GameState.FORTIFY):
            return self.__handle_fortify_phase(game_state)
        return None
