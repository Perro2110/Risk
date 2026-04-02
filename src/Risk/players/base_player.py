from abc import ABC, abstractmethod
from Risk.actions import Action, PlaceArmyAction, AttackAction
from Risk.game_state import GameState
from Risk.map import Country, Continent
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

    def set_game_state(self, game_state: GameState):
        self.game_state = game_state

    def choose_action(self) -> Action | None:
        """
        Choose the next action to execute, or None to end the current phase.

        Called repeatedly by `play_turn` until None is returned, at which
        point the game advances to the next phase.
        """
        if self.is_phase_applicable(GameState.PLACE_ARMY):
            return self.place_armies()

        elif self.is_phase_applicable(GameState.ATTACK):
            return self.attack()

        elif self.is_phase_applicable(GameState.FORTIFY):
            return self.fortify()

        return None

    @abstractmethod
    def place_armies(self) -> Action | None:
        """ The place army method used to place armies """
        pass

    @abstractmethod
    def attack(self) -> Action | None:
        """ The attack method used to attack """
        pass

    @abstractmethod
    def fortify(self) -> Action | None:
        """ The fortify method used to fortify """
        pass

    def turn_setup(self):
        """ Pre turn calculations in case a bot needs informations """
        pass

    def action_cleanup(self):
        """ Post turn calculations in case a bot needs to do something """
        pass

    def hog_wild_check(self):
        players = self.game_state.get_players()
        enemy_army_combined_size = 0

        for player in players:
            if player is self:
                continue

            enemy_army_combined_size += self.game_state.get_game_map() \
                                            .get_owned_army_size(player)

        self.is_attacking_hog_wild = self.game_state.get_game_map() \
            .get_owned_army_size(self) > enemy_army_combined_size

    def attack_hog_wild(self) -> Action | None:
        self.hog_wild_check()
        if self.is_attacking_hog_wild:
            self.attack_as_much_as_possible()

    def stalemate_check(self):
        self.is_stalemate = self.game_state.get_game_map() \
            .get_owned_army_size(self) > 300

    def attack_stalemate(self) -> Action | None:
        self.stalemate_check()
        if self.is_stalemate:
            self.attack_as_much_as_possible()

    def attack_easy_expand(
                self,
                root: list[Country] | Country
            ) -> Action | None:
        """
        Attack only when a border country has exactly one enemy neighbor and
        outnumbers it, expanding the cluster cleanly with minimal risk.

        For each border country of the given cluster, if it faces exactly one
        enemy neighbor and holds more armies than that enemy, it attacks and
        moves everything in to push the frontier forward.

        Args:
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
                root: list[Country] | Country
            ) -> Action | None:
        """
        Attack enemy countries that are completely surrounded by friendly
        territory, filling out gaps within the cluster to consolidate control.

        For each border country of the cluster, if an enemy neighbor has no
        enemy neighbors of its own (i.e. it is entirely surrounded by the
        player's countries), and the border country outnumbers it, attack and
        move zero armies in - keeping strength on the border.

        Args:
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
                        num_armies_to_attack = min(3, border.get_army_size()-1)
                        return AttackAction(
                            border,
                            enemy,
                            num_armies_to_attack,
                            0  # move zero in - keep strength on the border
                        )

        return None

    def attack_consolidate(
                self,
                root: list[Country] | Country
            ) -> Action | None:
        """
        Consolidate borders by coordinating attacks from multiple border
        countries into a single common enemy, reducing the number of exposed
        frontiers.

        For each border country with exactly one enemy neighbor, check if that
        enemy is bordered by more than one of the player's countries. If so,
        gather all friendly countries that also have only that single enemy
        neighbor and check whether their combined armies outnumber the target.
        If the odds are favorable, attack from each participant in turn, moving
        everything in to merge the frontlines.

        Args:
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

            # Get owned neighbors of the enemy that have only this one enemy
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
                cluster: list[Country],
                attack_ratio: float = 1.0
            ) -> Action | None:
        """
        Split a border country's armies across all its enemy neighbors when we
        sufficiently outnumber them, aggressively expanding in all directions.

        For each border country, sum the armies of all its enemy neighbors. If
        the border country armies exceed that total multiplied by attack_ratio,
        divide armies evenly across enemies and attack each one in turn, moving
        the per-enemy share in after each capture.

        Note: if armies are depleted early in the sequence of attacks, later
        enemies in the same border country's list may not be reached. This is
        an accepted limitation mirroring the original implementation.

        Args:
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
            enemy_armies_tot = sum(e.get_army_size() for e in enemy_neighbors)
            num_enemies = max(len(enemy_neighbors), 1)

            if border.get_army_size() > enemy_armies_tot * attack_ratio:
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

    def place_to_take_continent(self, continent: Continent) -> Action | None:
        # If we own the continent place 1 army at a time to the weakest border
        if continent.is_controlled_by(self):
            borders = continent.get_border_countries()

            if borders is None:
                return None

            country_to_place = utils.get_weakest_country(borders)

            if country_to_place is None:
                return None

            self.troops_to_place -= 1
            return PlaceArmyAction(
                country_to_place,
                1
            )

        countries = continent.get_owned_countries(self)

        # If we do no own the continent place all the armies to the most
        # contested country we own in the continent (if we own any)
        if len(countries) > 0:
            country_to_place = max(
                countries,
                key=lambda c: c.get_number_of_enemy_neighbors_in_cont(
                    continent,
                    self
                )
            )

            if country_to_place is None:
                return None

            troops_to_place = self.troops_to_place
            self.troops_to_place -= troops_to_place
            return PlaceArmyAction(
                country_to_place,
                troops_to_place
            )

        # If we do not own any country in the continent kys
        # TODO: don't
        return None

    def triple_attack_pack(self, root: Country) -> bool:
        """
        Run a combination of the three almost-always-helpful attacks on a
        single country root.

        Loops easy-expand until exhausted, runs one fill-out pass, then loops
        consolidate until exhausted.

        Returns:
            True if at least one attack was won, False otherwise.
        """
        won = False

        action = self.attack_easy_expand(root)
        while action is not None and self.has_won_last_attack:
            won = won or self.has_won_last_attack
            action.execute()
            action = self.attack_easy_expand(root)

        self.attack_fill_out(root)
        won = won or self.has_won_last_attack

        action = self.attack_consolidate(root)
        while action is not None and self.has_won_last_attack:
            won = won or self.has_won_last_attack
            action.execute()
            action = self.attack_consolidate(root)

        return won

    def attack_as_much_as_possible(self) -> Action | None:
        """
        Repeatedly attack with every owned country until no progress is made.

        Each pass iterates all owned countries, running `triple_attack_pack`
        and `attack_split_up` (with a near-zero ratio) on each. Continues
        until a full pass yields no successful attacks.
        """

        attacked = True
        while attacked:
            attacked = False
            owned_countries = self.game_state.get_game_map() \
                .get_owned_countries(self)

            for country in owned_countries:
                if self.triple_attack_pack(country):
                    attacked = True

                action = self.attack_split_up([country], attack_ratio=0.01)
                if action is not None:
                    action.execute()
                    attacked = True

    def add_completed_phase(self, phase: int):
        """Mark a phase as completed so it won't be re-entered this turn."""
        self.completed_phases.append(phase)

    def set_completed_phases(self, phases: list[int] | None = None):
        """Mark a phase as completed so it won't be re-entered this turn."""
        self.completed_phases = phases if phases is not None else []

    def set_troops_to_place(self, troops_to_place: int):
        """Override the number of troops available to place."""
        self.troops_to_place = troops_to_place

    def is_phase_applicable(self, phase: int) -> bool:
        """
        Return True if the game is in `phase` and it hasn't been completed yet.
        """
        return self.game_state.get_phase() == phase and \
            phase not in self.completed_phases

    def __str__(self) -> str:
        return self.color
