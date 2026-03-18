from Risk.actions import Action, PlaceArmyAction, AttackAction, FortifyAction
from Risk.game_state import GameState
from Risk import utils
from Risk.players.base_player import Player


class Angry(Player):
    """
    Angry strategy.

    An aggressive bot that maximises pressure on enemy territories:

    - **Place**: Drops all armies on the owned country with the most
      enemy neighbors.
    - **Attack**: Each owned country attacks its weakest enemy neighbor
      whenever it has numerical superiority and more than 1 army.
      After capturing a territory, armies are shifted toward the new
      frontline if it is more exposed than the source country.
    - **Fortify**: Pulls armies from lower-threat countries toward
      higher-threat friendly neighbors to reinforce the front.
    """

    def __init__(self, color: str, troops_to_place: int = 0):
        super().__init__(color, troops_to_place)
        self.placed_in_first = False

    def turn_setup(self):
        self.placed_in_first = False

    def place_armies(self) -> Action | None:
        """Place all available troops on the most contested owned country."""
        troops = self.troops_to_place
        if troops == 0:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        countries = self.game_state.get_game_map().get_owned_countries(self)
        sorted_countries = sorted(
            countries,
            key=lambda c: c.get_number_of_enemy_neighbors(self), reverse=True
        )

        if len(sorted_countries) == 0:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        if not self.placed_in_first:
            main = sorted_countries[0]
            main_troops = (self.troops_to_place * 2) // 3
            self.troops_to_place -= main_troops
            self.placed_in_first = True

            if len(sorted_countries) == 1:
                main_troops += self.troops_to_place
                self.troops_to_place = 0
                self.add_completed_phase(GameState.PLACE_ARMY)

            return PlaceArmyAction(main, main_troops)

        secondary = sorted_countries[1]
        self.troops_to_place -= troops
        self.add_completed_phase(GameState.PLACE_ARMY)
        return PlaceArmyAction(secondary, troops)

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
            weakest_en = utils.weakest_enemy_neighbour(country)

            if weakest_en is not None and \
                    country.get_army_size() > weakest_en.get_army_size() and \
                    country.get_army_size() > 1:

                num_armies_to_attack = min(3, country.get_army_size() - 1)

                # Default: keep armies at the source after the attack
                num_armies_want_to_move_post_attack = (
                    country.get_army_size() - 1 - num_armies_to_attack
                )

                # Move armies forward if the captured territory is more exposed
                if country.get_number_of_enemy_neighbors() - 1 > \
                        weakest_en.get_number_of_enemy_neighbors(self):
                    num_armies_want_to_move_post_attack = 0

                return AttackAction(
                    country,
                    weakest_en,
                    num_armies_to_attack,
                    num_armies_want_to_move_post_attack
                )

        return None  # No valid attacks found; end attack phase

    def fortify(self) -> Action | None:
        """
        Reinforce the front by moving armies from safer to more threatened
        connected friendly countries.
        """
        owned_countries = self.game_state.get_game_map() \
                                         .get_owned_countries(self)

        # Prioritise countries with the most enemy neighbors as destinations
        sorted_countries = sorted(
            owned_countries,
            key=lambda c: c.get_number_of_enemy_neighbors(),
            reverse=True
        )

        for to_country in sorted_countries:
            if not to_country.get_neighbors():
                continue

            for from_country in to_country.get_connected_friendly_countries():
                if from_country.get_army_size() <= 1:
                    continue

                num_armies_to_move = from_country.get_army_size() - 1
                self.add_completed_phase(GameState.FORTIFY)
                return FortifyAction(
                    from_country,
                    to_country,
                    num_armies_to_move
                )

        return None
