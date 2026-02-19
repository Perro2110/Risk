import random
from Risk.map import Map
from Risk.player import Player
from Risk.game_state import GameState
from Risk.visualizer import RiskVisualizer


class Game:
    """ Game class of Risk """

    def __init__(self, game_map: Map, game_length: int = 100):
        self.game_length = game_length  # number of turns before the game ends
        self.game_map = game_map
        self.players_list: list[Player] = []
        self.current_player = None
        self.game_state = None
        self.turn = 0

    def add_player(self, new_player: Player):
        self.players_list.append(new_player)

    def set_players(self, new_players_list: list[Player]):
        self.players_list = new_players_list

    def set_map(self, game_map: Map):
        """ Sets the game map """
        self.game_map = game_map

    def get_map(self) -> Map:
        """ Gets the game map """
        return self.game_map

    def get_game_state(self) -> GameState | None:
        """ Gets the game map """
        return self.game_state

    def init_game_state(self) -> None:
        self.game_state = GameState(
            self.game_map,
            [p.color for p in self.players_list]
        )

        num_player = len(self.players_list)
        num_starting_army = 50 - (5 * num_player)
        starting_army = [num_starting_army] * num_player

        # Randomly assign each country to a player
        countries_to_assign = self.get_map().get_countries()  # type: ignore
        random.shuffle(countries_to_assign)
        while len(countries_to_assign) > 0:
            for i in range(min(num_player, len(countries_to_assign))):
                c = countries_to_assign.pop()
                c.set_owner(self.players_list[i].color)
                c.set_army_size(1)
                starting_army[i] -= 1

        i = 0
        for player in self.players_list:
            player.set_troops_to_place(starting_army[i])
            action = player.choose_action(self.game_state)
            if action is not None:
                action.execute()

            i += 1

    def play(self):
        if self.get_map() is None:
            raise AttributeError('No game map found')

        if len(self.players_list) <= 1 or len(self.players_list) > 6:
            raise AttributeError('Invalid number of players')

        print('Initializing game state: ')
        self.init_game_state()
        print('Initial game state: ')
        print(self.game_state)

        print('Game starting: ')
        random.seed()

        viz = RiskVisualizer(self.get_game_state())  # type: ignore

        """ Plays the game until the end condition is met """
        while self.turn < self.game_length:
            print(f'Turn: {self.turn}')
            for player in self.players_list:
                self.current_player = player
                player.set_completed_phase([])
                print(f'player: {self.current_player.color}')
                player.play_turn(self.get_game_state(), viz)  # type: ignore
                self.game_state.next_player()  # type: ignore

            self.turn += 1

        viz.show()

        print('Final game state: ')
        for player in self.players_list:
            print(f"player {player.color} owns \
                {len(self.get_map().get_owned_countries(player.color))} \
            ")
