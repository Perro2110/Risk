import random
from Risk.map import Map
from Risk.player import Player
from Risk.game_state import GameState
from Risk.visualizer import RiskVisualizer
from matplotlib import pyplot as plt


class Game:
    """ Game class of Risk """

    def __init__(
                self,
                game_map: Map,
                game_length: int = 100,
                enable_visualizer: bool = True
            ):
        self.game_length = game_length  # number of turns before the game ends
        self.players: list[Player] = []
        self.game_state = GameState(game_map, [])
        self.viz = RiskVisualizer(self.get_game_state()) if enable_visualizer else None
        self.turn = 0

    def add_player(self, new_player: Player):
        self.players.append(new_player)
        self.get_game_state().set_players(self.players)  # type: ignore

    def set_players(self, new_players_list: list[Player]):
        self.players = new_players_list
        self.get_game_state().set_players(self.players)  # type: ignore

    def get_game_state(self) -> GameState:
        """ Gets the game map """
        return self.game_state

    def get_alive_players(self) -> list[Player]:
        return [p for p in self.players if not p.is_dead]

    def __init_game_state(self) -> None:
        num_player = len(self.players)
        num_starting_army = 50 - (5 * num_player)
        starting_army = [num_starting_army] * num_player

        # Randomly assign each country to a player
        free_countries = self.get_game_state().get_game_map().get_countries()
        random.shuffle(free_countries)
        while len(free_countries) > 0:
            for i in range(min(num_player, len(free_countries))):
                c = free_countries.pop()
                c.set_owner(self.players[i])
                c.set_army_size(1)
                starting_army[i] -= 1

        i = 0
        for player in self.players:
            player.set_troops_to_place(starting_army[i])
            while True:
                action = player.choose_action(self.game_state)
                if action is None:
                    break

                action.execute()

            i += 1

    def __play_turn(self, player: Player):
        """
        Drive a full turn: place armies → attack → fortify.

        Calls `choose_action` in a loop. Returning None advances the phase;
        the turn ends when phases wrap back to 0.
        """
        self.get_game_state().set_phase(GameState.PLACE_ARMY)
        player.set_troops_to_place(self.get_game_state()
                                   .get_reinforcements(player))
        while True:
            action = player.choose_action(self.get_game_state())

            if action is not None:
                action.execute()
                if self.viz is not None:
                    self.viz.show(False)
                    self.viz.update(self.get_game_state())
                    plt.pause(0.1)
            else:
                self.get_game_state().next_phase()
                if self.get_game_state().get_phase() == GameState.PLACE_ARMY:
                    return

    def __update_alive_players(self):
        game_map = self.get_game_state().get_game_map()
        for player in self.players:
            player.is_dead = len(game_map.get_owned_countries(player)) == 0

    def play(self):
        if len(self.players) <= 1 or len(self.players) > 6:
            raise AttributeError('Invalid number of players')

        print('Initializing game state: ')
        self.__init_game_state()
        print('Initial game state: ')
        print(self.game_state)

        print('Game starting: ')
        random.seed()

        """ Plays the game until the end condition is met """
        while self.turn < self.game_length:
            print(f'Turn: {self.turn}')

            for player in self.get_alive_players():
                player.set_completed_phases()
                print(f'player: {player}')
                self.__play_turn(player)
                self.get_game_state().next_player()

            self.__update_alive_players()
            self.turn += 1

        if self.viz is not None:
            self.viz.show()

        print('Final game state: ')
        for player in self.players:
            print(f"player {player} owns \
                {len(self.get_game_state().get_game_map()
                     .get_owned_countries(player))} \
            ")
