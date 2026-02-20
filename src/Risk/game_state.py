from Risk.map import Map


class GameState:
    """ Game state class """

    PLACE_ARMY: int = 0
    """ First phase of the turn, evaluates to 0 """

    ATTACK: int = 1
    """ Second phase of the turn, evaluates to 1 """

    FORTIFY: int = 2
    """ Third phase of the turn, evaluates to 2 """

    def __init__(
                self,
                game_map: Map,
                players: list[object],
            ):
        self.game_map = game_map
        self.players = {player.color: player for player in players}
        self.player_names = [player.color for player in players]
        self.current_player_index = 0
        self.phase = 0  # 0: place army, 1: attack, 2: fortify

    def get_game_map(self):
        return self.game_map

    def get_current_player(self):
        return self.players[self.player_names[self.current_player_index]]

    def get_alive_players(self):
        return [p for p in self.players.values() if not p.is_dead]

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
            current_player : {self.players[self.player_names[
                self.current_player_index
            ]]} \n \
            phase: {phases[self.phase]} \n \
            game_map: {self.game_map} \n \
        '
