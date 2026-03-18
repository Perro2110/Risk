import random
from Risk.map import Map
from Risk.players.base_player import Player
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
        self.enable_visualizer = enable_visualizer
        self.turn = 0

    def add_player(self, new_player: Player):
        if self.turn > 0:
            raise AttributeError("Cannot add player while game is running")

        self.players.append(new_player)
        self.get_game_state().set_players(self.players)  # type: ignore

    def set_players(self, new_players_list: list[Player]):
        if self.turn > 0:
            raise AttributeError("Cannot change players while game is running")

        self.players = new_players_list
        self.get_game_state().set_players(self.players)  # type: ignore

    def get_game_state(self) -> GameState:
        """ Gets the game state """
        return self.game_state

    def get_map(self) -> Map:
        """ Gets the game map """
        return self.get_game_state().get_game_map()

    def get_alive_players(self) -> list[Player]:
        return [p for p in self.players if not p.is_dead]

    def is_game_over(self) -> bool:
        """ Returns true if the game is over """
        return self.turn >= self.game_length \
            or len(self.get_alive_players()) == 1

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
            player.set_game_state(self.game_state)
            player.set_troops_to_place(starting_army[i])
            while True:
                action = player.choose_action()
                if action is None:
                    break

                action.execute()

            i += 1

        self.viz = RiskVisualizer(self.get_game_state()) \
            if self.enable_visualizer else None

    def __play_turn(self, player: Player):
        """
        Drive a full turn: place armies → attack → fortify.

        Calls `choose_action` in a loop. Returning None advances the phase;
        the turn ends when phases wrap back to 0.
        """
        self.get_game_state().set_phase(GameState.PLACE_ARMY)
        player.set_troops_to_place(
            self.get_game_state().get_reinforcements(player)
        )
        player.turn_setup()

        while True:
            action = player.choose_action()

            if action is not None:
                result = action.execute()
                player.action_cleanup()

                if isinstance(result, bool):
                    player.has_won_last_attack = result

                if self.viz is not None:
                    self.viz.show(False)
                    self.viz.update(self.get_game_state())
                    plt.pause(0.1)
            else:
                self.get_game_state().next_phase()
                player.action_cleanup()
                if self.get_game_state().get_phase() == GameState.PLACE_ARMY:
                    return

    def __update_alive_players(self):
        game_map = self.get_game_state().get_game_map()
        for player in self.get_alive_players():
            player.is_dead = len(game_map.get_owned_countries(player)) == 0
            if player.is_dead:
                self.get_game_state().add_to_leaderboard(player)

    def play(self, seed: int | None = None):
        if len(self.players) <= 1 or len(self.players) > 6:
            raise AttributeError('Invalid number of players')

        self.turn = 0

        print('Initializing game state: ')
        self.__init_game_state()
        print('Initial game state: ')
        print(self.game_state)

        print(f'Game starting{f' with seed {seed}' if seed else ''}: ')
        random.seed(seed)

        """ Plays the game until the end condition is met """
        while not self.is_game_over():
            print(f'Turn: {self.turn}')

            for player in self.get_alive_players():
                player.set_completed_phases()
                print(f'player: {player}')
                self.__play_turn(player)
                self.get_game_state().next_player()

            self.__update_alive_players()
            self.turn += 1

        if len(self.get_alive_players()) == 1:
            self.get_game_state().add_to_leaderboard(
                self.get_alive_players()[0]
            )
        else:
            # Sort players based on the number of controlled countries reversed
            sorted_players = sorted(
                self.get_alive_players(),
                key=lambda player: self.get_map().get_num_countries(player),
                reverse=True
            )

            for player in sorted_players:
                self.get_game_state().add_to_leaderboard(player)

        if self.viz is not None:
            self.viz.show()

        print('Final game state: ')
        for player in self.players:
            print(f"player {player} owns \
                {len(self.get_game_state().get_game_map()
                     .get_owned_countries(player))} \
            ")
