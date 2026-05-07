from __future__ import annotations

import random

from Risk.actions import Action, PlaceArmyAction, AttackAction, FortifyAction
from Risk.game_state import GameState
from Risk.map import Country, Continent
from Risk import utils
from Risk.players.base_player import Player


class SmartPlayer(Player):
    """
    Abstract base class for macro-driven smart bots.

    Provides a unified library of named macro-actions covering both the
    base place/attack/fortify strategies and the full strategy set of every
    regular (hand-coded) bot.  Concrete subclasses only need to implement
    `place_armies`, `attack`, and `fortify` - they call the `_execute_*_macro`
    helpers defined here to carry out the actual game logic.

    Macro naming convention
    -----------------------
    Base strategies:
        place_contested, place_weakest, place_continent
        attack_easy, attack_fill, attack_consolidate, attack_split, attack_pass
        fortify_border, fortify_pass

    Bot-mirrored strategies (named after the regular bot they replicate):
        place_angry,    place_stinky,    place_cluster,
        place_pixie,    place_communist
        attack_angry,   attack_stinky,   attack_communist,
        attack_pixie,   attack_cluster
        fortify_angry,  fortify_cluster, fortify_communist, fortify_pixie

    Architecture
    ------------
    Each _execute_*_macro(macro) wrapper calls the corresponding pure-logic
    helper _*_macro_action(macro) and manages phase completion.  Subclasses
    that handle phase completion themselves (e.g. DeepRLPH via _end_phase)
    should override the wrappers to call the raw helpers directly.

    Attributes:
        _cluster: Cached list of owned countries for attack macros; reset each
                turn in turn_setup.  Subclasses that override turn_setup
                must call super().turn_setup() to keep this fresh.
    """

    # ------------------------------------------------------------------
    # Macro catalogues
    # ------------------------------------------------------------------

    PLACE_MACROS: list[str] = [
        # base strategies
        "place_contested",
        "place_weakest",
        "place_continent",
        # bot-mirrored strategies
        "place_angry",
        "place_stinky",
        "place_cluster",
        "place_pixie",
        "place_communist",
    ]

    ATTACK_MACROS: list[str] = [
        # base strategies
        "attack_easy",
        "attack_fill",
        "attack_consolidate",
        "attack_split",
        "attack_pass",
        # bot-mirrored strategies
        "attack_angry",
        "attack_stinky",
        "attack_communist",
        "attack_pixie",
        "attack_cluster",
    ]

    FORTIFY_MACROS: list[str] = [
        # base strategies
        "fortify_border",
        "fortify_pass",
        # bot-mirrored strategies
        "fortify_angry",
        "fortify_cluster",
        "fortify_communist",
        "fortify_pixie",
    ]

    ALL_MACROS: list[str] = PLACE_MACROS + ATTACK_MACROS + FORTIFY_MACROS
    
    ALL_PHASES = {
        "place":   PLACE_MACROS,
        "attack":  ATTACK_MACROS,
        "fortify": FORTIFY_MACROS,
    }

    def __init__(self, color: str, troops_to_place: int = 0):
        super().__init__(color, troops_to_place)
        self._cluster: list[Country] | None = None

    def turn_setup(self):
        """Reset per-turn caches.  Subclasses must call super().turn_setup()."""
        self._cluster = None

    def _execute_place_macro(self, macro: str) -> Action | None:
        """Execute a place macro; marks phase complete when no action remains."""
        action = self._place_macro_action(macro)
        if action is None:
            self.add_completed_phase(GameState.PLACE_ARMY)
        return action

    def _execute_attack_macro(self, macro: str) -> Action | None:
        """Execute an attack macro; marks phase complete when no action remains."""
        action = self._attack_macro_action(macro)
        if action is None:
            self.add_completed_phase(GameState.ATTACK)
        return action

    def _execute_fortify_macro(self, macro: str) -> Action | None:
        """Execute a fortify macro; always marks phase complete (runs once)."""
        action = self._fortify_macro_action(macro)
        self.add_completed_phase(GameState.FORTIFY)
        return action


    def _place_macro_action(self, macro: str) -> Action | None:
        """
        Return the next PlaceArmyAction for the given macro, or None when done.

        One-at-a-time macros (place_contested, place_weakest, place_continent,
        place_cluster, place_communist) decrement troops_to_place by 1 and can
        be called repeatedly until None is returned.

        Dump-all macros (place_angry, place_stinky, place_pixie) place all
        remaining troops in a single call and set troops_to_place to 0.
        """
        owned = self.game_state.get_game_map().get_owned_countries(self)

        if self.troops_to_place <= 0 or not owned:
            return None

        target = None

        if macro == "place_contested":
            borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
            target = utils.get_most_contested_country(borders, self)

        elif macro in ("place_weakest", "place_communist"):
            # Communist places one at a time on the weakest country - same logic.
            target = utils.get_weakest_friendly_country(self.game_state, self)

        elif macro == "place_continent":
            best_cont, best_val = None, -1.0
            for cont in self.game_state.get_game_map().get_continents():
                owned_in = [c for c in cont.get_countries() if c in owned]
                n_total = len(cont.get_countries()) or 1
                progress = len(owned_in) / n_total
                val = progress * (1.0 + cont.get_reward(self) / 10.0)
                if val > best_val:
                    best_val, best_cont = val, cont
            if best_cont:
                action = self.place_to_take_continent(best_cont)
                if action:
                    return action
            # fall through to generic fallback

        elif macro == "place_angry":
            # Dump all remaining troops on the most contested border country.
            borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
            target = utils.get_most_contested_country(borders, self) or owned[0]
            troops = self.troops_to_place
            self.troops_to_place = 0
            return PlaceArmyAction(target, troops)

        elif macro == "place_stinky":
            # Dump all remaining troops on a random border country.
            borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
            target = random.choice(borders) if borders else owned[0]
            troops = self.troops_to_place
            self.troops_to_place = 0
            return PlaceArmyAction(target, troops)

        elif macro == "place_cluster":
            # Place 1 army on the weakest border of the best continent cluster.
            best_cont = self._get_best_cluster_continent()
            if best_cont:
                cluster = best_cont.get_best_cluster(self)
                if cluster:
                    borders = utils.get_cluster_borders(cluster)
                    if borders:
                        target = utils.get_weakest_country(borders)

        elif macro == "place_pixie":
            # Place to the continent where we have the best owned/enemy army ratio.
            best_cont, best_ratio = None, -1.0
            for cont in self.game_state.get_game_map().get_continents():
                owned_army = cont.get_owned_army_size(self)
                enemy_army = cont.get_enemy_army_size(self) or 1
                ratio = owned_army / enemy_army
                if ratio > best_ratio:
                    best_ratio, best_cont = ratio, cont
            if best_cont:
                action = self.place_to_take_continent(best_cont)
                if action:
                    return action
            # fall through to generic fallback

        # ── generic fallback (most contested border, then first owned) ────────
        if target is None:
            borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
            target = (
                utils.get_most_contested_country(borders, self)
                or owned[0]
            )

        self.troops_to_place -= 1
        return PlaceArmyAction(target, 1)

    def _attack_macro_action(self, macro: str) -> Action | None:
        """
        Return the next AttackAction for the given macro, or None when done.

        A None return signals that the attack phase should end (no valid attack
        is available, or attack_pass was selected).  The _cluster cache is
        initialised lazily on the first attack call each turn and reset by
        turn_setup.
        """
        if macro == "attack_pass":
            return None

        if self._cluster is None:
            self._cluster = (
                self.game_state.get_game_map().get_owned_countries(self)
            )

        if macro == "attack_easy":
            return self.attack_easy_expand(self._cluster)

        elif macro == "attack_fill":
            return self.attack_fill_out(self._cluster)

        elif macro == "attack_consolidate":
            return self.attack_consolidate(self._cluster)

        elif macro == "attack_split":
            return self.attack_split_up(self._cluster, attack_ratio=1.2)

        elif macro in ("attack_angry", "attack_communist"):
            # Attack the weakest enemy neighbor whenever we strictly outnumber it.
            for country in self._cluster:
                weakest = utils.weakest_enemy_neighbour(country)
                if (weakest is not None
                        and country.get_army_size() > weakest.get_army_size()
                        and country.get_army_size() > 1):
                    num = min(3, country.get_army_size() - 1)
                    post = country.get_army_size() - 1 - num
                    if (country.get_number_of_enemy_neighbors() - 1
                            > weakest.get_number_of_enemy_neighbors(self)):
                        post = 0
                    return AttackAction(country, weakest, num, post)
            return None

        elif macro == "attack_stinky":
            # Attack the weakest enemy neighbor only when we have a 1.5x edge.
            for country in self._cluster:
                weakest = utils.weakest_enemy_neighbour(country)
                if (weakest is not None
                        and country.get_army_size()
                            > weakest.get_army_size() * 1.5
                        and country.get_army_size() > 1):
                    num = min(3, country.get_army_size() - 1)
                    post = country.get_army_size() - 1 - num
                    next_weakest = utils.weakest_enemy_neighbour(weakest)
                    if (next_weakest is None
                            or weakest.get_army_size()
                                > next_weakest.get_army_size()):
                        post = 0
                    return AttackAction(country, weakest, num, post)
            return None

        elif macro == "attack_pixie":
            # Continent-focused: attack weakest in-continent enemy when weaker.
            gm = self.game_state.get_game_map()
            for continent in gm.get_continents():
                all_cont = continent.get_countries()
                borders = continent.get_bordering_countries(self) or []
                countries = continent.get_owned_countries(self) + borders
                for c in countries:
                    if c.get_army_size() <= 1:
                        continue
                    enemies_in_cont = [
                        n for n in c.get_enemy_neighbors(self)
                        if n in all_cont
                    ]
                    weakest = utils.get_weakest_country(enemies_in_cont)
                    if (weakest is not None
                            and weakest.get_army_size() < c.get_army_size()):
                        num = min(3, c.get_army_size() - 1)
                        post = c.get_army_size() - 1 - num
                        if (c.get_number_of_enemy_neighbors() - 1
                                > weakest.get_number_of_enemy_neighbors(self)):
                            post = 0
                        return AttackAction(c, weakest, num, post)
            return None

        elif macro == "attack_cluster":
            # Easy-expand restricted to the best continent cluster's borders.
            best_cont = self._get_best_cluster_continent()
            if best_cont:
                cluster = best_cont.get_best_cluster(self)
                if cluster:
                    borders = utils.get_cluster_borders(cluster)
                    if borders:
                        return self.attack_easy_expand(borders)
            return None

        return None

    def _fortify_macro_action(self, macro: str) -> Action | None:
        """
        Return a single FortifyAction for the given macro, or None.

        Fortify executes at most once per turn; the caller always marks the
        phase complete regardless of whether an action was found.
        """
        if macro == "fortify_pass":
            return None

        gm = self.game_state.get_game_map()
        owned = gm.get_owned_countries(self)

        if not owned:
            return None

        if macro == "fortify_border":
            # Most contested border <- richest interior connected country.
            borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
            if not borders:
                return None
            target = utils.get_most_contested_country(borders, self)
            if not target:
                return None
            interior = [
                c for c in target.get_connected_friendly_countries()
                if c.get_number_of_enemy_neighbors() == 0
                and c.get_army_size() > 1
            ]
            if not interior:
                return None
            source = max(interior, key=lambda c: c.get_army_size())
            n = source.get_army_size() - 1
            return FortifyAction(source, target, n) if n > 0 else None

        elif macro == "fortify_angry":
            # Most threatened dest <- first reachable connected source.
            for to_c in sorted(
                owned,
                key=lambda c: c.get_number_of_enemy_neighbors(),
                reverse=True,
            ):
                if not to_c.get_neighbors():
                    continue
                for from_c in to_c.get_connected_friendly_countries():
                    if from_c.get_army_size() <= 1:
                        continue
                    return FortifyAction(from_c, to_c, from_c.get_army_size() - 1)
            return None

        elif macro == "fortify_cluster":
            # Equalize armies across the best continent cluster's borders.
            best_cont = self._get_best_cluster_continent()
            if not best_cont:
                return None
            cluster = best_cont.get_best_cluster(self)
            if not cluster:
                return None
            borders = utils.get_cluster_borders(cluster)
            if not borders:
                return None
            mean = sum(c.get_army_size() for c in borders) / len(borders)
            for to_c in sorted(borders, key=lambda c: c.get_army_size() - mean):
                if not to_c.get_neighbors():
                    continue
                for from_c in sorted(
                    cluster,
                    key=lambda c: c.get_army_size() - mean,
                    reverse=True,
                ):
                    if from_c.get_army_size() <= 1:
                        break
                    n = (from_c.get_army_size() - to_c.get_army_size()) // 2
                    if n <= 0:
                        continue
                    return FortifyAction(from_c, to_c, n)
            return None

        elif macro == "fortify_communist":
            # Equalize armies across all owned countries.
            mean = sum(c.get_army_size() for c in owned) / len(owned)
            for to_c in sorted(owned, key=lambda c: c.get_army_size() - mean):
                if not to_c.get_neighbors():
                    continue
                for from_c in sorted(
                    to_c.get_connected_friendly_countries(),
                    key=lambda c: c.get_army_size() - mean,
                    reverse=True,
                ):
                    if from_c.get_army_size() <= 1:
                        break
                    n = (from_c.get_army_size() - to_c.get_army_size()) // 2
                    if n <= 0:
                        continue
                    return FortifyAction(from_c, to_c, n)
            return None

        elif macro == "fortify_pixie":
            # Per-continent: weakest border <- richest reachable interior.
            for continent in gm.get_continents():
                cont_owned = continent.get_owned_countries(self)
                if not cont_owned:
                    continue
                borders = utils.get_cluster_borders(cont_owned)
                if not borders:
                    continue
                weakest_border = utils.get_weakest_country(borders)
                if not weakest_border:
                    continue
                reachable = weakest_border.get_connected_friendly_countries()
                interior = [
                    c for c in reachable
                    if c.get_number_of_enemy_neighbors() == 0
                    and c.get_army_size() > 1
                    and c is not weakest_border
                ]
                if not interior:
                    continue
                from_c = max(interior, key=lambda c: c.get_army_size())
                n = from_c.get_army_size() - 1
                if n > 0:
                    return FortifyAction(from_c, weakest_border, n)
            return None

        return None

    def _get_best_cluster_continent(self) -> Continent | None:
        """
        Return the continent with the largest best-cluster within a fair-share
        size budget (same heuristic as the Cluster bot).
        """
        gm = self.game_state.get_game_map()
        n_players = max(len(self.game_state.get_players()), 1)
        max_size = len(gm.get_countries()) // n_players
        best_size = 0
        best_cont: Continent | None = None
        for cont in gm.get_continents():
            if len(cont.get_countries()) > max_size:
                continue
            cluster = cont.get_best_cluster(self)
            if cluster is None:
                continue
            if len(cluster) > best_size:
                best_size = len(cluster)
                best_cont = cont
        return best_cont