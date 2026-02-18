from Risk.map import Map
from Risk.player import Player


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

    def add_player(self, new_player: Player):
        self.players_list.append(new_player)

    def set_players(self, new_players_list: list[Player]):
        self.players_list = new_players_list
        

    def set_map(self, game_map: Map):
        """ Sets the game map """
        self.game_map = game_map

    def get_map(self) -> Map | None:
        """ Gets the game map """
        return self.game_map

    def play(self):
        """ Plays the game until the end condition is met """
        while self.turn < self.game_length:
            for player in self.players_list:
                self.current_player = player
                action = player.choose_action(self.get_game_state())
                if action is not None:
                    action.execute()
            self.turn += 1
