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

    def __str__(self):
        phases = ['place army', 'attack', 'fortify']
        return f'\
            current_player : {self.players[self.current_player_index]} \n \
            phase: {phases[self.phase]} \n \
            game_map: {self.game_map} \n \
        '