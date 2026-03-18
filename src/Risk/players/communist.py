from Risk.actions import Action, PlaceArmyAction, AttackAction, FortifyAction
from Risk.game_state import GameState
from Risk import utils
from Risk.players.base_player import Player


class Communist(Player):
    """
        Comunist strategy (stub, always passes).
        Communist uses the concept that all countries are equal. Communist will
        place its armies one by one, placing each one in the weakest country.
        The bot will attack the weakest neighbor it has on the board, but only
        if it has more armies than the opponent. When it comes to moving or
        fortification the goal is that every country should have as many armies
        as its friendly neighbors. Another feature that Communist has is that
        it will not attack another player if it also is a Communist. At least
        not until there are only Communist players left in the game. It might
        seem as if Communist violates the set rule that bots should not
        cooperation. This will only be the case if there is more than one
        Communist in a particular game.
    """

    def __init__(self, color: str, troops_to_place: int = 0):
        super().__init__(color, troops_to_place)
        self.strategy = 'communist'

    def place_armies(self) -> Action | None:
        troops = self.troops_to_place
        if troops == 0:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        country_to_place = utils.get_weakest_friendly_country(
            self.game_state,
            self
        )

        if country_to_place is None:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        self.troops_to_place -= 1
        return PlaceArmyAction(country_to_place, 1)

    def attack(self) -> Action | None:
        """
        Attack with every eligible country.

        A country is eligible if it has a weaker enemy neighbor and holds
        more than 1 army. Up to 3 armies attack; post-attack movement
        favors the newly captured country if it is more exposed.
        """
        owned_countries = self.game_state.get_game_map() \
                                         .get_owned_countries(self)

        for country in owned_countries:
            weakest_en = utils.weakest_enemy_neighbour_list(country)

            if weakest_en is None:
                continue

            for enemy in weakest_en:
                if country.get_army_size() >= enemy.get_army_size() and \
                        country.get_army_size() > 1:

                    num_armies_to_attack = min(3, country.get_army_size() - 1)

                    # Default: keep armies at the source after the attack
                    troops_to_move_post_attack = (country.get_army_size() - 1
                                                  - num_armies_to_attack) // 2

                    return AttackAction(
                        country,
                        enemy,
                        num_armies_to_attack,
                        troops_to_move_post_attack
                    )

        self.add_completed_phase(GameState.ATTACK)
        return None  # No valid attacks found; end attack phase

    def fortify(self) -> Action | None:
        """
        RedPlayer equalizes the owned countries armies by finding the country
        with the least troops and the one with the most and splitting the army
        equally between the two
        """
        owned_countries = self.game_state.get_game_map() \
                                         .get_owned_countries(self)

        if len(owned_countries) == 0:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        mean_army_amount = sum([c.get_army_size() for c in owned_countries]) \
            / len(owned_countries)

        sorted_countries = sorted(
            owned_countries,
            key=lambda c: c.get_army_size() - mean_army_amount
        )

        for to_country in sorted_countries:
            if not to_country.get_neighbors():
                continue

            possible_from_countries = sorted(
                to_country.get_connected_friendly_countries(),
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
