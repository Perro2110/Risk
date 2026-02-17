from Risk.map import Map

class GameState:
    """ Game state class """

    def __init__(
            self,
            game_map: Map,
            players: list[str],
            current_player_index: int = 0,
            phase: int = 0
        ):
        self.game_map = game_map
        self.players = players
        self.current_player_index = 0
        self.phase = 0  # 0: place army, 1: attack, 2: fortify

    def get_game_map(self):
        return self.game_map

    def get_current_player(self):
        return self.players[self.current_player_index]

    def next_player(self):
        self.current_player_index = (
            self.current_player_index + 1
        ) % len(self.players)

    def get_phase(self):
        return self.phase

    def set_phase(self, phase):
        self.phase = phase

    def next_phase(self):
        self.phase = (self.phase + 1) % 3


class Game:
    """ Game class of Risk (Singleton) """

    instance = None

    def __init__(self, game_length: int = 100, game_map: Map | None = None):
        self.game_length = game_length  # number of turns before the game ends
        self.game_map = game_map
        self.players_list = []
        self.current_player = None
        self.turn = 0

    # def __new__(cls):
    #     """
    #     Override __new__ to implement the Singleton pattern.
    #     """
    #     if cls.instance is None:
    #         cls.instance = super().__new__(cls)
    #     return cls.instance

    def set_map(self, game_map: Map):
        """ Sets the game map """
        self.game_map = game_map

    def get_map(self) -> Map | None:
        """ Gets the game map """
        return self.game_map

    def play(self):
        """ Plays the game until the end condition is met (game_length turns)"""
        # while self.turn < self.game_length:
        #     for player in self.players_list:
        #         self.current_player = player
        #         action = player.choose_action(self.get_game_state())
        #         if action is not None:
        #             action.execute()
        #     self.turn += 1