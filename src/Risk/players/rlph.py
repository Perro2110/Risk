from __future__ import annotations
import random
import json
from collections import defaultdict
from Risk.actions import Action, PlaceArmyAction, FortifyAction
from Risk.game_state import GameState
from Risk.map import Country
from Risk import utils
from Risk.players.base_player import Player


class RLPH(Player):
    """
    Risk Learning PowerHouse. (0% winrate btw) TODO: change accordingly
    Player che impara a giocare a Risk tramite Q-learning tabulare.

    Sceglie tra macro-azioni (strategie ad alto livello già implementate
    in Player) invece di singole mosse, mantenendo lo spazio stati/azioni
    abbastanza piccolo da convergere in ~500 partite.

    Args:
        color:   colore del player (come per tutti gli altri player)
        alpha:   learning rate (default 0.1)
        gamma:   discount factor (default 0.9)
        epsilon: tasso di esplorazione epsilon-greedy (default 0.2)
                 metti 0.0 per usare solo la Q-table appresa
    """

    PLACE_MACROS = [
        "place_contested",   # rinforza il confine più conteso
        "place_weakest",     # rinforza il paese più debole
        "place_continent",   # rinforza il continente più promettente
    ]

    ATTACK_MACROS = [
        "attack_easy",         # espandi dove sei sicuro di vincere
        "attack_fill",         # elimina isole nemiche circondate
        "attack_consolidate",  # riduci i fronti aperti
        "attack_split",        # attacca in più direzioni contemporaneamente
        "attack_pass",         # passa (fine fase attacco)
    ]

    FORTIFY_MACROS = [
        "fortify_border",    # sposta truppe verso il confine più esposto
        "fortify_pass",      # passa
    ]

    ALL_MACROS = PLACE_MACROS + ATTACK_MACROS + FORTIFY_MACROS

    def __init__(
        self,
        color: str,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        troops_to_place: int = 0,
    ):
        super().__init__(color, troops_to_place)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q[stato][macro] = valore stimato
        self.Q: dict[tuple, dict[str, float]] = defaultdict(
            lambda: {m: 0.0 for m in self.ALL_MACROS}
        )

        self._phase_macro: dict[int, str] = {}
        self._prev_state: tuple | None = None
        self._prev_macro: str | None = None
        self._prev_countries: int = 0
        self._prev_armies: int = 0

        self._cluster: list[Country] | None = None

    def _extract_state(self) -> tuple:
        """
        Ritorna una tupla discreta che rappresenta la situazione corrente.

        Feature:
            0. army_ratio      - armate proprie / armate totali
            1. country_ratio   - paesi propri / paesi totali
            2. continent_bonus - rinforzi da continenti (buckettato)
            3. border_pressure - media nemici per paese di confine
            4. phase           - fase di gioco (0/1/2)
        """
        def _bucket(value: float, thresholds: list[float]) -> int:
            for i, t in enumerate(thresholds):
                if value <= t:
                    return i
            return len(thresholds)

        game_map = self.game_state.get_game_map()
        all_countries = game_map.get_countries()
        owned = game_map.get_owned_countries(self)

        total_armies = sum(c.get_army_size() for c in all_countries) or 1
        own_armies = sum(c.get_army_size() for c in owned)

        total_countries = len(all_countries) or 1

        continent_bonus = game_map.get_reward(self)

        borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
        border_pressure = (
            sum(
                c.get_number_of_enemy_neighbors() for c in borders
            ) / len(borders) if borders else 0.0
        )

        players = self.game_state.get_players()
        enemy_max_countries = max(
            (
                len(game_map.get_owned_countries(p))
                for p in players if p is not self
            ), default=1
        )
        threat = len(owned) / (enemy_max_countries or 1)

        return (
            _bucket(own_armies / total_armies,     [0.15, 0.30, 0.50, 0.70]),
            _bucket(len(owned) / total_countries,  [0.15, 0.30, 0.50, 0.70]),
            _bucket(continent_bonus,               [3, 6, 9, 12]),
            _bucket(border_pressure,               [1.0, 2.0, 3.0, 4.0]),
            _bucket(threat,                        [0.5, 0.8, 1.2, 2.0]),
        )

    def _compute_reward(self) -> float:
        game_map = self.game_state.get_game_map()
        n_countries = len(game_map.get_owned_countries(self))
        n_armies = game_map.get_owned_army_size(self)

        delta_c = (n_countries - self._prev_countries) * 3.0
        delta_a = (n_armies - self._prev_armies) * 0.1

        self._prev_countries = n_countries
        self._prev_armies = n_armies

        return delta_c + delta_a

    def _update_q(self, reward: float, next_state: tuple):
        if self._prev_state is None or self._prev_macro is None:
            return
        best_next = max(self.Q[next_state].values())
        td_target = reward + self.gamma * best_next
        self.Q[self._prev_state][self._prev_macro] += (
            self.alpha * (
                td_target - self.Q[self._prev_state][self._prev_macro]
            )
        )

    def action_cleanup(self):
        self._record_transition()

    def _record_transition(self):
        next_state = self._extract_state()
        reward = self._compute_reward()
        self._update_q(reward, next_state)
        self._prev_state = next_state

    def _select_macro(self, candidates: list[str]) -> str:
        if random.random() < self.epsilon:
            return random.choice(candidates)
        state = self._extract_state()
        q_vals = {m: self.Q[state][m] for m in candidates}
        return max(q_vals, key=q_vals.__getitem__)

    def _execute_place_macro(self, macro: str) -> Action | None:
        owned = self.game_state.get_game_map().get_owned_countries(self)

        if self.troops_to_place <= 0:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        target = None

        if macro == "place_contested":
            borders = [
                c for c in owned if c.get_number_of_enemy_neighbors() > 0
            ]
            target = utils.get_most_contested_country(borders, self)

        elif macro == "place_weakest":
            target = utils.get_weakest_friendly_country(self.game_state, self)

        elif macro == "place_continent":
            best_cont, best_val = None, -1.0
            for cont in self.game_state.get_game_map().get_continents():
                val = cont.get_owned_army_size(self) \
                    / (cont.get_enemy_army_size(self) or 1)
                if val > best_val:
                    best_val, best_cont = val, cont
            if best_cont:
                action = self.place_to_take_continent(best_cont)
                if action:
                    return action

        # fallback generico se la macro non ha trovato un target
        if target is None:
            borders = [c for c in owned if c.get_number_of_enemy_neighbors()]
            target = utils.get_most_contested_country(borders, self) \
                or (owned[0] if owned else None)

        if target is None:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        self.troops_to_place -= 1
        return PlaceArmyAction(target, 1)

    def _execute_attack_macro(self, macro: str) -> Action | None:
        if macro == "attack_pass":
            self.add_completed_phase(GameState.ATTACK)
            return None

        if self._cluster is None:
            self._cluster = self.game_state.get_game_map()\
                .get_owned_countries(self)

        fallback_order = [macro, "attack_easy", "attack_fill",
                          "attack_consolidate", "attack_split"]
        seen = set()
        for m in fallback_order:
            if m in seen:
                continue
            seen.add(m)
            if m == "attack_easy":
                action = self.attack_easy_expand(self._cluster)
            elif m == "attack_fill":
                action = self.attack_fill_out(self._cluster)
            elif m == "attack_consolidate":
                action = self.attack_consolidate(self._cluster)
            elif m == "attack_split":
                action = self.attack_split_up(self._cluster, attack_ratio=1.2)
            else:
                continue
            if action is not None:
                return action

        self.add_completed_phase(GameState.ATTACK)
        return None

    def _execute_fortify_macro(self, macro: str) -> Action | None:
        if macro == "fortify_pass":
            self.add_completed_phase(GameState.FORTIFY)
            return None

        owned = self.game_state.get_game_map().get_owned_countries(self)
        borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]

        if not borders:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        target = utils.get_most_contested_country(borders, self)

        if not target:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        connected = target.get_connected_friendly_countries()
        interior = [
            c for c in connected if c.get_number_of_enemy_neighbors() == 0
            and c.get_army_size() > 1
        ]

        if not interior:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        source = max(interior, key=lambda c: c.get_army_size())
        n = source.get_army_size() - 1

        if n <= 0:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        self.add_completed_phase(GameState.FORTIFY)
        return FortifyAction(source, target, n)

    def turn_setup(self):
        self._cluster = None
        self._phase_macro: dict[int, str] = {}  # one macro decision per phase
        game_map = self.game_state.get_game_map()
        self._prev_countries = len(game_map.get_owned_countries(self))
        self._prev_armies = game_map.get_owned_army_size(self)
        self._prev_state = self._extract_state()

    def _get_phase_macro(self, phase: int, candidates: list[str]) -> str:
        """
        Return the macro locked for this phase this turn.
        Selects once at phase entry, then reuses for every subsequent call
        within the same phase — so Q-updates are attributed to one decision,
        not to dozens of individual army placements.
        """
        if phase not in self._phase_macro:
            self._phase_macro[phase] = self._select_macro(candidates)
        return self._phase_macro[phase]

    def place_armies(self) -> Action | None:
        macro = self._get_phase_macro(GameState.PLACE_ARMY, self.PLACE_MACROS)
        self._prev_macro = macro
        return self._execute_place_macro(macro)

    def attack(self) -> Action | None:
        macro = self._get_phase_macro(GameState.ATTACK, self.ATTACK_MACROS)
        self._prev_macro = macro
        return self._execute_attack_macro(macro)

    def fortify(self) -> Action | None:
        macro = self._get_phase_macro(GameState.FORTIFY, self.FORTIFY_MACROS)
        self._prev_macro = macro
        return self._execute_fortify_macro(macro)

    def receive_terminal_reward(self, won: bool):
        """Chiama questo metodo una volta alla fine di ogni partita."""
        reward = 50.0 if won else -50.0
        if self._prev_state is not None and self._prev_macro is not None:
            self._update_q(reward, self._extract_state())

    def save(self, filepath: str):
        """Salva la Q-table su file JSON."""
        data = {str(state): values for state, values in self.Q.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[RLPlayer {self.color}] Q-table salvata in {filepath} ({
                len(self.Q)
            } stati)")

    def load(self, filepath: str):
        """Carica una Q-table salvata in precedenza."""
        with open(filepath, "r", encoding="utf-8") as f:
            raw: dict[str, dict[str, float]] = json.load(f)
        self.Q = defaultdict(lambda: {m: 0.0 for m in self.ALL_MACROS})
        for state_str, values in raw.items():
            state = tuple(int(x) for x in state_str.strip("()").split(","))
            self.Q[state] = values
        print(f"[RLPlayer {self.color}] Q-table caricata da {filepath} ({
                len(self.Q)
            } stati)")
