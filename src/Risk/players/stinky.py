import random
from Risk.actions import Action, PlaceArmyAction, AttackAction
from Risk.game_state import GameState
from Risk import utils
from Risk.players.base_player import Player


class Stinky(Player):
    """Green player - Stinky strategy (stub, always passes)."""

    def place_armies(self) -> Action | None:
        troops = self.troops_to_place
        if troops == 0:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        all_owned = self.game_state.get_game_map().get_owned_countries(self)

        if len(all_owned) == 0:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        possible_countries = [
            n for n in all_owned
            if n.get_number_of_enemy_neighbors() > 0
        ]

        random.shuffle(possible_countries)

        country_to_place = possible_countries[0]

        if country_to_place is None:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        self.troops_to_place -= troops
        self.add_completed_phase(GameState.PLACE_ARMY)
        return PlaceArmyAction(country_to_place, troops)

    def attack(self) -> Action | None:
        """
        Attack with every eligible country.

        A country is eligible if it has a weaker enemy neighbor and holds
        more than 1 army. Up to 3 armies attack; post-attack movement
        favors the newly captured country if it is more exposed.
        """
        owned_countries = self.game_state.get_game_map() \
                                         .get_owned_countries(self)

        is_army_huge = sum([
            c.get_army_size() for c in owned_countries
        ]) > 500

        for country in owned_countries:
            weakest_en = utils.weakest_enemy_neighbour(country)

            if weakest_en is not None and \
                    (country.get_army_size() >
                     weakest_en.get_army_size() * 1.5 or is_army_huge) \
                    and country.get_army_size() > 1:

                num_armies_to_attack = min(3, country.get_army_size() - 1)

                # Default: keep armies at the source after the attack
                num_armies_want_to_move_post_attack = (
                    country.get_army_size() - 1 - num_armies_to_attack
                )

                next_weakest = utils.weakest_enemy_neighbour(weakest_en)

                # Move armies forward if the captured territory is more exposed
                if next_weakest is None or weakest_en.get_army_size() > \
                        next_weakest.get_army_size():
                    num_armies_want_to_move_post_attack = 0

                return AttackAction(
                    country,
                    weakest_en,
                    num_armies_to_attack,
                    num_armies_want_to_move_post_attack
                )

        return None  # No valid attacks found; end attack phase

    def fortify(self) -> Action | None:
        return None
