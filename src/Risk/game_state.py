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
                players: list[object]
            ):
        self.game_map = game_map
        self.players = players
        self.leaderboard = []
        self.current_player_index = 0
        self.phase = 0  # 0: place army, 1: attack, 2: fortify

    def set_game_map(self, game_map: Map):
        """ Add a game map to the game state """
        self.game_map = game_map

    def get_game_map(self):
        """ Get the game map """
        return self.game_map

    def get_reinforcements(self, player):
        """ Returns the number of placeable troops for the selected player """
        return self.get_game_map().get_reward(player)

    def get_current_player(self):
        """ Get the current player """
        return self.players[self.current_player_index]

    def set_players(self, players: list[object]):
        """ Sets the game players """
        self.players = players

    def get_players(self):
        """ Get the game players """
        return self.players

    def next_player(self):
        """ Changes the current player to the next in the player order """
        next_index = self.current_player_index + 1
        self.current_player_index = next_index % len(self.players)

    def add_to_leaderboard(self, player: object):
        """ Add the player to the start of the leaderboard """
        self.leaderboard.insert(0, player)

    def get_leaderboard(self):
        """ Return the leaderboard """
        return self.leaderboard

    def get_phase(self) -> int:
        """ Get the current game phase """
        return self.phase

    def set_phase(self, phase: int):
        """ Sets the game phase """
        if phase > self.FORTIFY or phase < self.PLACE_ARMY:
            raise ValueError("Invalid game phase")

        self.phase = phase

    def next_phase(self):
        """ Changes the current phase to the next in the phase order """
        self.phase = (self.phase + 1) % 3

    def __str__(self):
        phases = ['place army', 'attack', 'fortify']
        return f'\
            current_player : {self.players[self.current_player_index]} \n \
            phase: {phases[self.phase]} \n \
            game_map: {self.game_map} \n \
        '
