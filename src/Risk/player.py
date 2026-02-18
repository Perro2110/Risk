from abc import ABC, abstractmethod
from Risk.actions import Action, PlaceArmyAction, AttackAction, FortifyAction
from Risk.game_state import GameState
from Risk import utils


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
    """ Player base class """

    def __init__(self, color: str, troops_to_place: int = 0):
        self.color = color
        self.troops_to_place = troops_to_place

    @abstractmethod
    def choose_action(self, game_state: GameState) -> Action | None:
        """
        This method is called during the player's turn to choose an action
        to execute. The player can choose to execute an action or to end
        their turn (by returning None).
        """
        pass

    def set_troops_to_place(self, troops_to_place: int):
        self.troops_to_place = troops_to_place

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
    def play_turn(self, game_state: GameState):
        game_state.set_phase(0)  # start with place army phase
        self.troops_to_place = game_state.get_game_map().get_reward(self.color)
        while True:
            action = self.choose_action(game_state)
            if action is not None:
                action.execute()
            else:
                game_state.next_phase()
                if game_state.get_phase() == 0:
                    return


class RedPlayer(Player):
    """ Red player class (Comunist strategy — stub) """

    def choose_action(self, game_state: GameState) -> Action | None:
        """
        Stub implementation — always ends the turn.
        To be implemented with a more sophisticated strategy.
        """
        return None


class PurplePlayer(Player):
    """ Purple player class (Pixie strategy — stub) """

    def choose_action(self, game_state: GameState) -> Action | None:
        """
        Stub implementation — always ends the turn.
        To be implemented with a more sophisticated strategy.
        """
        return None


class YellowPlayer(Player):
    """ Yellow player class (Cluster strategy — stub) """

    def choose_action(self, game_state: GameState) -> Action | None:
        """
        Stub implementation — always ends the turn.
        To be implemented with a more sophisticated strategy.
        """
        return None


class GreenPlayer(Player):
    """ Green player class (Stinky strategy — stub) """

    def choose_action(self, game_state: GameState) -> Action | None:
        """
        Stub implementation — always ends the turn.
        To be implemented with a more sophisticated strategy.
        """
        return None


class BluePlayer(Player):
    """ Blue player class (Neferius strategy — stub) """

    def choose_action(self, game_state: GameState) -> Action | None:
        """
        Stub implementation — always ends the turn.
        To be implemented with a more sophisticated strategy.
        """
        return None


class BlackPlayer(Player):
    """
    Black player class — Angry strategy.

    Angry is an aggressive bot:
    - Place phase: puts all armies on the country with the most enemy neighbors
    - Attack phase: every owned country attacks its weakest enemy neighbor,
      but only if the attacker has more armies than the target and more than 1.
      If it takes a country all armies will be moved to the country which has 
      the most enemy neighbors.

    - Fortify phase: moves armies from countries with fewer enemy neighbors
      to neighboring friendly countries that face more enemies.
    """

    def choose_action(self, game_state: GameState) -> Action | None:
        if game_state.get_phase() == 0:
            # Place army phase
            troops = self.troops_to_place
            if self.troops_to_place == 0:
                return None

            country_to_place = utils.get_most_contested_country(
                game_state,
                self.color
            )

            if country_to_place is None:
                return None

            self.troops_to_place -= troops
            return PlaceArmyAction(country_to_place, self.troops_to_place)

        elif game_state.get_phase() == 1:
            # Attack phase
            owned_countries = game_state.get_game_map() \
                .get_owned_countries(self.color)

            for country in owned_countries:
                weakest_en = utils.weakest_enemy_neighbour(country)

                if weakest_en is not None and \
                        country.get_army_size() > weakest_en.get_army_size() \
                        and country.get_army_size() > 1:

                    num_armies_to_attack = min(3, country.get_army_size() - 1)
                    
                    num_armies_want_to_move_post_attack = (
                            country.get_army_size() - 1 - num_armies_to_attack
                    )

                    if country.get_number_of_enemy_neighbors() - 1 \
                            > weakest_en.get_number_of_enemy_neighbors(
                                self.color
                            ):
                        num_armies_want_to_move_post_attack = 0

                    return AttackAction(
                        country,
                        weakest_en,
                        num_armies_to_attack,
                        num_armies_want_to_move_post_attack
                    )

            return None

        elif game_state.get_phase() == 2:
            # Fortify phase
            owned_countries = game_state.get_game_map() \
                .get_owned_countries(self.color)

            for country in owned_countries:
                country_to_place = utils.get_most_contested_neighbour(country)

                if country_to_place is not None and \
                        country_to_place.get_number_of_enemy_neighbors() \
                        > country.get_number_of_enemy_neighbors():
                    num_armies_to_fortify = country.get_army_size() - 1
                    return FortifyAction(
                        country,
                        country_to_place,
                        num_armies_to_fortify
                    )
                            
            return None
