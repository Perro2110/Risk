from Risk.actions import Action, PlaceArmyAction, FortifyAction
from Risk.game_state import GameState
from Risk import utils
from Risk.players.base_player import Player


class Cluster(Player):
    """ Cluster strategy. """

    def __init__(self, color: str, troops_to_place: int = 0):
        super().__init__(color, troops_to_place)
        self.best_continent = None

    def turn_setup(self):
        """ Find the best continent by cluster size """
        max_size = len(self.game_state.get_game_map().get_countries()) \
            // len(self.game_state.get_players())
        continents = self.game_state.get_game_map().get_continents()
        best_cluster = 0
        best_continent = None

        for continent in continents:
            if len(continent.get_countries()) > max_size:
                continue

            cluster = continent.get_best_cluster(self)

            if cluster is None:
                continue

            cluster_size = len(cluster)
            if cluster_size > best_cluster:
                best_cluster = cluster_size
                best_continent = continent

        self.borders = None
        self.has_won_last_attack = True
        self.best_continent = best_continent
        self.done_attack_step = [False] * 5

    def place_armies(self) -> Action | None:
        """
        Uniformly split the armies between the borders of the best cluster.
        """
        troops = self.troops_to_place
        if troops == 0:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        if self.best_continent is None:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        if self.borders is None:
            cluster = self.best_continent.get_best_cluster(self)

            if cluster is None:
                self.add_completed_phase(GameState.PLACE_ARMY)
                return None

            borders = utils.get_cluster_borders(cluster)

            if borders is None:
                self.add_completed_phase(GameState.PLACE_ARMY)
                return None

            self.borders = borders

        country_to_place = utils.get_weakest_country(self.borders)

        if country_to_place is None:
            country_to_place = utils.get_most_contested_country(
                self.game_state.get_game_map().get_owned_countries(self), self
            )

        if country_to_place is None:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        self.troops_to_place -= 1
        return PlaceArmyAction(country_to_place, 1)

    def attack(self) -> Action | None:
        """
        Bunch of different attack logic implemented in base Player class.
        """
        if self.borders is None:
            self.add_completed_phase(GameState.ATTACK)
            return None

        borders = self.borders
        if not self.done_attack_step[0]:
            if self.has_won_last_attack:
                action = self.attack_easy_expand(borders)
                if action is not None:
                    action.execute()
            self.done_attack_step[0] = True
            self.has_won_last_attack = True

        if not self.done_attack_step[1]:
            self.done_attack_step[1] = True

            action = self.attack_fill_out(borders)
            if action is not None:
                action.execute()

            self.has_won_last_attack = True

        if not self.done_attack_step[2]:
            if self.has_won_last_attack:
                action = self.attack_easy_expand(borders)
                if action is not None:
                    action.execute()
            self.done_attack_step[2] = True
            self.has_won_last_attack = True

        if not self.done_attack_step[3]:
            if self.has_won_last_attack:
                action = self.attack_consolidate(borders)
                if action:
                    action.execute()
            self.done_attack_step[3] = True
            self.has_won_last_attack = True

        if not self.done_attack_step[4]:
            if self.has_won_last_attack:
                action = self.attack_split_up(borders, 1)
                if action is not None:
                    action.execute()
            self.done_attack_step[4] = True

        self.add_completed_phase(GameState.ATTACK)
        return None

    def fortify(self) -> Action | None:
        """
        Sposta da dove ha più truppe verso il bordo del cluster con meno truppe
        bilanciando le armate dei 2 paesi
        """
        if self.best_continent is None:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        cluster = self.best_continent.get_best_cluster(self)

        if cluster is None:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        borders = utils.get_cluster_borders(cluster)

        if borders is None:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        mean_army_amount = sum([c.get_army_size() for c in borders]) \
            / len(borders)

        sorted_countries = sorted(
            borders,
            key=lambda c: c.get_army_size() - mean_army_amount
        )

        for to_country in sorted_countries:
            if not to_country.get_neighbors():
                continue

            possible_from_countries = sorted(
                cluster,
                key=lambda c: c.get_army_size() - mean_army_amount,
                reverse=True
            )

            for from_country in possible_from_countries:
                if from_country.get_army_size() <= 1:
                    break

                num_armies_to_move = (from_country.get_army_size()
                                      - to_country.get_army_size()) // 2

                self.add_completed_phase(GameState.FORTIFY)
                return FortifyAction(
                    from_country,
                    to_country,
                    num_armies_to_move
                )
