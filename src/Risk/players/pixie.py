from Risk.actions import Action, PlaceArmyAction, AttackAction, FortifyAction
from Risk.game_state import GameState
from Risk import utils
from Risk.players.base_player import Player


class Pixie(Player):
    """Purple player - Pixie strategy (stub, always passes)."""

    def __init__(self, color: str, troops_to_place: int = 0):
        super().__init__(color, troops_to_place)
        self.chance_to_own = []
        self.continent_needs_help = True
        self.place_country_id = 0
        self.wants_a_continent = False
        self.continents = []

    def turn_setup(self):
        """ Find the best continent by cluster size """
        if self.continents is None:
            self.continents = self.game_state.get_game_map().get_continents()

        best_ratio = 0
        best_continent = None

        for continent in self.continents:
            owned_army = continent.get_owned_army_size(self)
            enemy_army = continent.get_enemy_army_size(self)

            current_ratio = owned_army / enemy_army

            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_continent = continent

        self.wants_a_continent = False
        self.continent_needs_help = True
        self.calculate_wanted_continents(self.troops_to_place)
        self.place_country_id = 0
        self.borders = None
        self.has_won_last_attack = True
        self.best_continent = best_continent
        self.done_attack_step = [False] * 5

    def calculate_wanted_continents(self, troops_to_place: int = 1):
        contient_number = len(self.continents)
        if self.chance_to_own is None:
            self.chance_to_own = [False] * contient_number

        needed_army = [0] * contient_number
        for i in range(contient_number):
            needed_army[i] += self.continents[i].get_enemy_army_size(self)
            needed_army[i] -= self.continents[i].get_owned_army_size(self)

            bordering_countries = self.continents[i] \
                .get_bordering_countries(self)

            if bordering_countries is not None:
                needed_army[i] -= sum([
                    c.get_army_size() for c in bordering_countries
                ])

        for i in range(contient_number):
            if needed_army[i] < troops_to_place/contient_number:
                self.wants_a_continent = True
                self.chance_to_own[i] = True
            else:
                self.chance_to_own[i] = False

    def place_armies(self) -> Action | None:
        # If we want a continent place it in the best possible continent
        # This is calculated at turn start in the turn_setup function
        if self.wants_a_continent and self.best_continent:
            return self.place_to_take_continent(
                self.best_continent
            )

        # If we do not want any continent we cycle through all of them
        # placing in groups of 1 only in continents that we have a chance
        # to own (self.chance_to_own[i]) and it needs help
        # (check function for more details)
        if self.continent_needs_help:
            self.continent_needs_help = False

            for (i, continent) in enumerate(self.continents):
                # TODO: check
                if self.chance_to_own[i] and \
                        utils.continent_needs_help(self, continent):
                    self.troops_to_place -= 1
                    self.continent_needs_help = True

                    return self.place_to_take_continent(continent)

        # jenky ahh fix to keep track of the next country to place 1 troop in
        owned = self.game_state.get_game_map().get_owned_countries(self)
        if len(owned) == 0:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        current_country = owned[self.place_country_id]

        if self.troops_to_place > 0:
            if current_country.get_number_of_enemy_neighbors() > 0:
                self.troops_to_place -= 1
                self.place_country_id += 1
                if self.place_country_id == len(owned):
                    self.place_country_id = 0

                return PlaceArmyAction(
                    current_country,
                    1
                )

        self.add_completed_phase(GameState.PLACE_ARMY)
        return None

    def attack(self) -> Action | None:
        # For each continent, if we have a chance to own it, we attack just
        # just like black player except we attack only neighbors inside the
        # continent
        for (i, continent) in enumerate(self.continents):
            if self.chance_to_own[i]:
                all_cont_countries = continent.get_countries()
                borders = continent.get_bordering_countries(self)
                borders = borders if borders is not None else []

                countries = continent.get_owned_countries(self) + borders

                for c in countries:
                    if c.get_army_size() <= 1:
                        continue

                    enemies_in_cont = c.get_enemy_neighbors(self)
                    enemies_in_cont = [
                        n for n in enemies_in_cont
                        if n in all_cont_countries
                    ]

                    weakest_en = utils.get_weakest_country(enemies_in_cont)

                    if weakest_en is not None and \
                            weakest_en.get_army_size() < c.get_army_size():

                        num_armies_to_attack = min(3, c.get_army_size() - 1)

                        # Default: keep armies at the source after the attack
                        num_armies_want_to_move_post_attack = (
                            c.get_army_size() - 1 - num_armies_to_attack
                        )

                        # Move armies forward if the captured territory is
                        # more exposed
                        if c.get_number_of_enemy_neighbors() - 1 > \
                                weakest_en.get_number_of_enemy_neighbors(self):
                            num_armies_want_to_move_post_attack = 0

                        AttackAction(
                            c,
                            weakest_en,
                            num_armies_to_attack,
                            num_armies_want_to_move_post_attack
                        ).execute()

        # TODO: check
        # self.attack_hog_wild()
        # self.attack_stalemate()

        self.add_completed_phase(GameState.ATTACK)
        return None

    def fortify(self) -> Action | None:
        """
        Continent-focused fortification.

        Priority 1 - for each continent Purple wants to own, find interior
        countries (no enemy neighbors) and funnel their armies toward the
        weakest border country in that continent to reinforce the front.

        Priority 2 - fallback across all owned countries: move armies from
        the least contested reachable country to the most contested one,
        so spare troops are never stranded in safe territory.
        """
        # Priority 1: reinforce contested continent border
        for i, continent in enumerate(self.continents):
            if not self.chance_to_own[i]:
                continue

            borders = utils.get_cluster_borders(
                continent.get_owned_countries(self)
            )
            if not borders:
                continue

            weakest_border = utils.get_weakest_country(borders)
            if weakest_border is None:
                continue

            # Look for interior countries with spare armies that can reach
            # the weakest border through friendly territory
            reachable = weakest_border.get_connected_friendly_countries()
            interiors = [
                c for c in reachable
                if c.get_number_of_enemy_neighbors() == 0
                and c.get_army_size() > 1
                and c is not weakest_border
            ]

            if not interiors:
                continue

            # Drain the richest interior country toward the weakest border
            from_country = max(interiors, key=lambda c: c.get_army_size())
            num_armies_to_move = from_country.get_army_size() - 1

            self.add_completed_phase(GameState.FORTIFY)
            return FortifyAction(
                from_country,
                weakest_border,
                num_armies_to_move
            )

        # Priority 2: global fallback - safest country → most threatened
        owned = self.game_state.get_game_map().get_owned_countries(self)

        # Most threatened reachable destination
        threatened = sorted(
            owned,
            key=lambda c: c.get_number_of_enemy_neighbors(),
            reverse=True
        )

        for to_country in threatened:
            if to_country.get_number_of_enemy_neighbors() == 0:
                break  # no point reinforcing safe countries

            reachable = to_country.get_connected_friendly_countries()
            candidates = [
                c for c in reachable
                if c.get_army_size() > 1
                and c.get_number_of_enemy_neighbors()
                < to_country.get_number_of_enemy_neighbors()
            ]

            if not candidates:
                continue

            from_country = min(
                candidates,
                key=lambda c: c.get_number_of_enemy_neighbors()
            )
            num_armies_to_move = (
                from_country.get_army_size() - to_country.get_army_size()
            ) // 2

            if num_armies_to_move <= 0:
                continue

            self.add_completed_phase(GameState.FORTIFY)
            return FortifyAction(from_country, to_country, num_armies_to_move)

        self.add_completed_phase(GameState.FORTIFY)
        return None
