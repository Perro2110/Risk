"""
players.py — Player classes for the Risk game engine.

Each player subclass implements a distinct strategy via `choose_action`.
The base `Player` class handles turn flow (place → attack → fortify phases).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from Risk.actions import Action, PlaceArmyAction, AttackAction, FortifyAction
from Risk.game_state import GameState
from Risk.map import Country
from Risk import utils


class Player(ABC):
    """
    Abstract base class for all Risk players.

    Manages turn state (current phase, troops remaining) and drives the
    game loop via `play_turn`. Concrete subclasses must implement
    `choose_action` to define their strategy.

    Attributes:
        color (str): The player's color identifier.
        troops_to_place (int): Armies remaining to place in the current turn.
        completed_phases (list[int]): Phases already finished this turn.
    """

    def __init__(self, color: str, troops_to_place: int = 0):
        self.color = color
        self.strategy = None
        self.is_dead = False
        self.is_attacking_hog_wild = False
        self.is_stalemate = False
        self.has_won_last_attack = False
        self.troops_to_place = troops_to_place
        self.completed_phases: list[int] = []

    def choose_action(self, game_state: GameState) -> Action | None:
        """
        Choose the next action to execute, or None to end the current phase.

        Called repeatedly by `play_turn` until None is returned, at which
        point the game advances to the next phase.
        """
        if self.is_phase_applicable(game_state, GameState.PLACE_ARMY):
            return self.place_armies(game_state)

        elif self.is_phase_applicable(game_state, GameState.ATTACK):
            return self.attack(game_state)

        elif self.is_phase_applicable(game_state, GameState.FORTIFY):
            return self.fortify(game_state)

        return None

    @abstractmethod
    def place_armies(self, game_state: GameState) -> Action | None:
        """ The place army method used to place armies """
        pass

    @abstractmethod
    def attack(self, game_state: GameState) -> Action | None:
        """ The attack method used to attack """
        pass

    @abstractmethod
    def fortify(self, game_state: GameState) -> Action | None:
        """ The fortify method used to fortify """
        pass

    def turn_setup(self, game_state: GameState):
        """ Pre turn calculations in case a bot needs informations """
        pass

    def hog_wild_check(self, game_state: GameState):
        players = game_state.get_players()
        enemy_army_combined_size = 0

        for player in players:
            if player is self:
                continue

            enemy_army_combined_size += game_state.get_game_map() \
                                                  .get_owned_army_size(player)

        self.is_attacking_hog_wild = game_state.get_game_map() \
            .get_owned_army_size(self) > enemy_army_combined_size

    def attack_hog_wild(self, game_state: GameState) -> Action | None:
        if self.is_attacking_hog_wild:
            self.attack_as_much_as_possible(game_state)

    def stalemate_check(self, game_state: GameState):
        self.is_stalemate = game_state.get_game_map() \
            .get_owned_army_size(self) > 500

    def attack_stalemate(self, game_state: GameState) -> Action | None:
        if self.is_stalemate:
            self.attack_as_much_as_possible(game_state)

    def attack_easy_expand(
                self,
                game_state: GameState,
                root: list[Country] | Country
            ) -> Action | None:
        """
        Attack only when a border country has exactly one enemy neighbor and
        outnumbers it, expanding the cluster cleanly with minimal risk.

        For each border country of the given cluster, if it faces exactly one
        enemy neighbor and holds more armies than that enemy, it attacks and
        moves everything in to push the frontier forward.

        Args:
            game_state: The current game state.
            cluster: The list of countries forming the cluster whose borders
                    will be evaluated for easy-expand attacks.

        Returns:
            An AttackAction if a valid easy-expand opportunity is found,
            or None if no such attack exists.
        """
        borders = utils.get_cluster_borders(root)
        if borders is None:
            return None

        for border in borders:
            enemy_neighbors = border.get_enemy_neighbors(self)
            if len(enemy_neighbors) == 1:
                enemy = enemy_neighbors[0]
                if border.get_army_size() > enemy.get_army_size():
                    num_armies_to_attack = min(3, border.get_army_size() - 1)
                    troops_to_move_post_attack = border.get_army_size() - 1 \
                        - num_armies_to_attack
                    return AttackAction(
                        border,
                        enemy,
                        num_armies_to_attack,
                        troops_to_move_post_attack
                    )

        return None

    def attack_fill_out(
                self,
                game_state: GameState,
                root: list[Country] | Country
            ) -> Action | None:
        """
        Attack enemy countries that are completely surrounded by friendly territory,
        filling out gaps within the cluster to consolidate control.

        For each border country of the cluster, if an enemy neighbor has no
        enemy neighbors of its own (i.e. it is entirely surrounded by the
        player's countries), and the border country outnumbers it, attack and
        move zero armies in — keeping strength on the border.

        Args:
            game_state: The current game state.
            cluster: The list of countries forming the cluster whose borders
                    will be evaluated for fill-out attacks.

        Returns:
            An AttackAction if a valid fill-out opportunity is found,
            or None if no such attack exists.
        """
        borders = utils.get_cluster_borders(root)
        if borders is None:
            return None

        for border in borders:
            for enemy in border.get_enemy_neighbors(self):
                if len(enemy.get_enemy_neighbors(self)) == 0:
                    if border.get_army_size() > enemy.get_army_size():
                        num_armies_to_attack = min(3, border.get_army_size() - 1)
                        return AttackAction(
                            border,
                            enemy,
                            num_armies_to_attack,
                            0  # move zero in — keep strength on the border
                        )

        return None

    def attack_consolidate(
                self,
                game_state: GameState,
                root: list[Country] | Country
            ) -> Action | None:
        """
        Consolidate borders by coordinating attacks from multiple border countries
        into a single common enemy, reducing the number of exposed frontiers.

        For each border country with exactly one enemy neighbor, check if that
        enemy is bordered by more than one of the player's countries. If so,
        gather all friendly countries that also have only that single enemy
        neighbor and check whether their combined armies outnumber the target.
        If the odds are favorable, attack from each participant in turn, moving
        everything in to merge the frontlines.

        Args:
            game_state: The current game state.
            cluster: The list of countries forming the cluster whose borders
                    will be evaluated for consolidation attacks.

        Returns:
            An AttackAction if a valid consolidation attack is found,
            or None if no such opportunity exists.
        """
        borders = utils.get_cluster_borders(root)
        if borders is None:
            return None

        for border in borders:
            enemy_neighbors = border.get_enemy_neighbors(self)

            if len(enemy_neighbors) != 1:
                continue

            enemy = enemy_neighbors[0]

            # Enemy must be bordered by more than one of our countries
            if enemy.get_number_of_friendly_neighbors(self) <= 1:
                continue

            # Get all friendly neighbors of the enemy that have only this one enemy
            participants = [
                n for n in enemy.get_friendly_neighbors(self)
                if n.get_number_of_enemy_neighbors() == 1
            ]

            our_armies = sum(c.get_army_size() for c in participants)
            if our_armies <= enemy.get_army_size():
                continue

            # Attack from each participant until the enemy is conquered
            for attacker in participants:
                if enemy.get_owner() is self:
                    break
                if attacker.get_army_size() > 1 \
                        and attacker.is_neighbor(enemy):
                    num_armies_to_attack = min(3, attacker.get_army_size() - 1)
                    troops_to_move_post_attack = attacker.get_army_size() - 1 \
                        - num_armies_to_attack
                    return AttackAction(
                        attacker,
                        enemy,
                        num_armies_to_attack,
                        troops_to_move_post_attack
                    )

        return None

    def attack_split_up(
                self,
                game_state: GameState,
                cluster: list[Country],
                attack_ratio: float = 1.0
            ) -> Action | None:
        """
        Split a border country's armies across all its enemy neighbors when
        we sufficiently outnumber them, aggressively expanding in all directions.

        For each border country, sum the armies of all its enemy neighbors. If
        the border country's armies exceed that total multiplied by attack_ratio,
        divide armies evenly across enemies and attack each one in turn, moving
        the per-enemy share in after each capture.

        Note: if armies are depleted early in the sequence of attacks, later
        enemies in the same border country's list may not be reached. This is
        an accepted limitation mirroring the original implementation.

        Args:
            game_state: The current game state.
            cluster: The list of countries forming the cluster whose borders
                    will be evaluated for split-up attacks.
            attack_ratio: Multiplier applied to total enemy armies to set the
                        threshold for attacking. 1.0 means we must at least
                        match them; higher values require greater superiority.

        Returns:
            An AttackAction if a valid split-up attack is found,
            or None if no such opportunity exists.
        """
        borders = utils.get_cluster_borders(cluster)
        if borders is None:
            return None

        for border in borders:
            enemy_neighbors = border.get_enemy_neighbors(self)
            enemy_armies_total = sum(e.get_army_size() for e in enemy_neighbors)
            num_enemies = max(len(enemy_neighbors), 1)

            if border.get_army_size() > enemy_armies_total * attack_ratio:
                armies_per = border.get_army_size() // num_enemies

                for enemy in enemy_neighbors:
                    if border.get_army_size() <= 1:
                        break
                    num_armies_to_attack = min(3, border.get_army_size() - 1)
                    armies_per = min(armies_per, border.get_army_size() - 1)
                    return AttackAction(
                        border,
                        enemy,
                        num_armies_to_attack,
                        armies_per
                    )

        return None

    def attack_as_much_as_possible(
                self,
                game_state: GameState
            ) -> Action | None:

        # TODO: add

        pass

    def add_completed_phase(self, phase: int):
        """Mark a phase as completed so it won't be re-entered this turn."""
        self.completed_phases.append(phase)

    def set_completed_phases(self, phases: list[int] = []):
        """Mark a phase as completed so it won't be re-entered this turn."""
        self.completed_phases = phases

    def set_troops_to_place(self, troops_to_place: int):
        """Override the number of troops available to place."""
        self.troops_to_place = troops_to_place

    def is_phase_applicable(self, game_state: GameState, phase: int) -> bool:
        """
        Return True if the game is in `phase` and it hasn't been completed yet.
        """
        return game_state.get_phase() == phase and \
            phase not in self.completed_phases

    def __str__(self) -> str:
        return self.color


class RedPlayer(Player):
    """
        Red player — Comunist strategy (stub, always passes).
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

    def place_armies(self, game_state: GameState) -> Action | None:
        troops = self.troops_to_place
        if troops == 0:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        country_to_place = utils.get_weakest_friendly_country(game_state, self)

        if country_to_place is None:
            return None

        self.troops_to_place -= 1
        return PlaceArmyAction(country_to_place, 1)

    def attack(self, game_state: GameState) -> Action | None:
        """
        Attack with every eligible country.

        A country is eligible if it has a weaker enemy neighbor and holds
        more than 1 army. Up to 3 armies attack; post-attack movement
        favors the newly captured country if it is more exposed.
        """
        owned_countries = game_state.get_game_map().get_owned_countries(self)

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

                    self.add_completed_phase(GameState.ATTACK)

                    return AttackAction(
                        country,
                        enemy,
                        num_armies_to_attack,
                        troops_to_move_post_attack
                    )

        return None  # No valid attacks found; end attack phase

    def fortify(self, game_state: GameState) -> Action | None:
        """
        TODO:
        """
        owned_countries = game_state.get_game_map().get_owned_countries(self)

        # TODO:
        if len(owned_countries) == 0:
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


class PurplePlayer(Player):
    """Purple player — Pixie strategy (stub, always passes)."""

    def choose_action(self, game_state: GameState) -> Action | None:
        return None


class YellowPlayer(Player):
    """Yellow player — Cluster strategy (stub, always passes)."""

    def __init__(self, color: str, troops_to_place: int = 0):
        super().__init__(color, troops_to_place)
        self.best_continent = None

    def turn_setup(self, game_state: GameState):
        """ Trovare il suo continente migliore """
        max_size = len(game_state.get_game_map().get_countries()) \
            // len(game_state.get_players())
        continents = game_state.get_game_map().get_continents()
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

    def place_armies(self, game_state: GameState) -> Action | None:
        """
        Mette le truppe uniformemente sul bordo del suo cluster nel continente
        """
        troops = self.troops_to_place
        if troops == 0:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        if self.best_continent is None:
            return None

        if self.borders is None:
            cluster = self.best_continent.get_best_cluster(self)

            if cluster is None:
                return None

            borders = utils.get_cluster_borders(cluster)

            if borders is None:
                return None

            self.borders = borders

        country_to_place = utils.get_weakest_country(self.borders)

        if country_to_place is None:
            return None

        self.troops_to_place -= 1
        return PlaceArmyAction(country_to_place, 1)

    def attack(self, game_state: GameState) -> Action | None:
        """
        I bordi del cluster nel continente attaccano i vicini con meno truppe
        Come black player sposta dopo attacco se il paese che conquista ha
        vicini più deboli
        """
        if self.borders is None:
            return None

        borders = self.borders
        if not self.done_attack_step[0]:
            if self.has_won_last_attack:
                action = self.attack_easy_expand(game_state, borders)
                if action is not None:
                    return action
            self.done_attack_step[0] = True
            self.has_won_last_attack = True

        if not self.done_attack_step[1]:
            self.done_attack_step[1] = True

            action = self.attack_fill_out(game_state, borders)
            if action is not None:
                return action

            self.has_won_last_attack = True

        if not self.done_attack_step[2]:
            if self.has_won_last_attack:
                action = self.attack_easy_expand(game_state, borders)
                if action is not None:
                    return action
            self.done_attack_step[2] = True
            self.has_won_last_attack = True

        if not self.done_attack_step[3]:
            if self.has_won_last_attack:
                action = self.attack_consolidate(game_state, borders)
                if action is not None:
                    return action
            self.done_attack_step[3] = True
            self.has_won_last_attack = True

        if not self.done_attack_step[4]:
            if self.has_won_last_attack:
                action = self.attack_split_up(game_state, borders, 1.2)
                if action is not None:
                    return action
            self.done_attack_step[4] = True

        self.add_completed_phase(GameState.ATTACK)
        return None

    def fortify(self, game_state: GameState) -> Action | None:
        """
        Sposta da dove ha più truppe verso il bordo del cluster con meno truppe
        bilanciando le armate dei 2 paesi
        """
        if self.best_continent is None:
            return None

        cluster = self.best_continent.get_best_cluster(self)

        if cluster is None:
            return None

        borders = utils.get_cluster_borders(cluster)

        if borders is None:
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


class GreenPlayer(Player):
    """Green player — Stinky strategy (stub, always passes)."""

    def choose_action(self, game_state: GameState) -> Action | None:
        return None


class BluePlayer(Player):
    """Blue player — Neferius strategy (stub, always passes)."""

    def choose_action(self, game_state: GameState) -> Action | None:
        return None


class BlackPlayer(Player):
    """
    Black player — Angry strategy.

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

    def place_armies(self, game_state: GameState) -> Action | None:
        """Place all available troops on the most contested owned country."""
        troops = self.troops_to_place
        if troops == 0:
            return None

        country_to_place = utils.get_most_contested_country(game_state, self)
        if country_to_place is None:
            return None

        self.troops_to_place -= troops
        self.add_completed_phase(GameState.PLACE_ARMY)
        return PlaceArmyAction(country_to_place, troops)

    def attack(self, game_state: GameState) -> Action | None:
        """
        Attack with every eligible country.

        A country is eligible if it has a weaker enemy neighbor and holds
        more than 1 army. Up to 3 armies attack; post-attack movement
        favors the newly captured country if it is more exposed.
        """
        owned_countries = game_state.get_game_map().get_owned_countries(self)

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

    def fortify(self, game_state: GameState) -> Action | None:
        """
        Reinforce the front by moving armies from safer to more threatened
        connected friendly countries.
        """
        owned_countries = game_state.get_game_map().get_owned_countries(self)

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
