from __future__ import annotations

import random
import json
from collections import defaultdict

from Risk.actions import Action, PlaceArmyAction, FortifyAction
from Risk.game_state import GameState
from Risk.map import Country
from Risk import utils
from Risk.players.base_player import Player


# ---------------------------------------------------------------------------
# RLPH+ — Risk Learning, versione potenziata
# ---------------------------------------------------------------------------
# Differenze rispetto a RLPH originale:
#
#   FIX 1 — la fase di gioco (place/attack/fortify) entra nello stato.
#            Senza questo, la stessa tupla stato veniva usata per aggiornare
#            macro di fasi diverse — la Q-table non riusciva a distinguerle.
#
#   FIX 2 — reward ribilanciata:
#            · delta_a scende da 0.5 a 0.1 (le armate cambiano troppo spesso)
#            · aggiunto continent_gain (+5 per ogni punto bonus conquistato)
#            · aggiunta front_penalty (-0.1 per paese di confine aperto)
#
#   FIX 3 — _record_transition() chiamato dopo ogni singola azione
#            (place_armies / attack / fortify), non una sola volta a fine turno
#            tramite action_cleanup. Il segnale arriva subito, non aggregato.
#
#   FIX 4 — epsilon decade del 0.5% dopo ogni partita, floor a 0.05.
#            Prima era fisso per sempre e continuava ad esplorare casualmente
#            anche dopo 1000 partite.
#
#   FIX 5 — attack_pass penalizzato se c'erano attacchi disponibili.
#            Prima poteva essere rinforzato per caso se il reward del turno
#            era positivo per altri motivi.
#
#   FIX 6 — turn_setup inizializza _prev_cont_bonus.
#            Prima il primo delta continenti del turno era spazzatura.
# ---------------------------------------------------------------------------


# ── costanti ────────────────────────────────────────────────────────────────
ALPHA         = 0.1    # learning rate
GAMMA         = 0.9    # discount factor
EPSILON_START = 0.25   # esplorazione iniziale
EPSILON_MIN   = 0.05   # floor esplorazione
EPSILON_DECAY = 0.995  # moltiplicatore per partita  [FIX 4]

# indici di fase — usati nel bucket dello stato  [FIX 1]
PHASE_PLACE   = 0
PHASE_ATTACK  = 1
PHASE_FORTIFY = 2


class RLPHPlus(Player):
    """
    Q-learning tabulare con macro-azioni, versione corretta.

    Args:
        color           colore del player
        alpha           learning rate           (default 0.1)
        gamma           discount factor         (default 0.9)
        epsilon         esplorazione iniziale   (default 0.25)
        troops_to_place truppe iniziali
    """

    PLACE_MACROS = [
        "place_contested",
        "place_weakest",
        "place_continent",
    ]
    ATTACK_MACROS = [
        "attack_easy",
        "attack_fill",
        "attack_consolidate",
        "attack_split",
        "attack_pass",
    ]
    FORTIFY_MACROS = [
        "fortify_border",
        "fortify_pass",
    ]
    ALL_MACROS = PLACE_MACROS + ATTACK_MACROS + FORTIFY_MACROS

    def __init__(
        self,
        color: str,
        alpha:   float = ALPHA,
        gamma:   float = GAMMA,
        epsilon: float = EPSILON_START,
        troops_to_place: int = 0,
    ):
        super().__init__(color, troops_to_place)
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon

        # Q[stato][macro] = valore stimato
        self.Q: dict[tuple, dict[str, float]] = defaultdict(
            lambda: {m: 0.0 for m in self.ALL_MACROS}
        )

        # memoria di transizione
        self._prev_state:      tuple | None = None
        self._prev_macro:      str   | None = None
        self._prev_countries:  int          = 0
        self._prev_armies:     int          = 0
        self._prev_cont_bonus: float        = 0.0   # [FIX 6]
        self._current_phase:   int          = PHASE_PLACE  # [FIX 1]

        self._cluster: list[Country] | None = None
        self._games_played: int = 0

    # ── estrazione stato ────────────────────────────────────────────────────
    def _extract_state(self) -> tuple:
        """
        Tupla discreta che rappresenta la situazione corrente.

        Feature (rispetto a RLPH originale, aggiunta la fase):
            0  army_ratio        armate proprie / totale  (bucket 5 livelli)
            1  country_ratio     paesi propri / totale    (bucket 5 livelli)
            2  continent_bonus   bonus continenti         (bucket 5 livelli)
            3  border_pressure   media nemici per confine (bucket 5 livelli)
            4  phase             0=place  1=attack  2=fortify  [FIX 1]
        """
        def _bucket(value: float, thresholds: list[float]) -> int:
            for i, t in enumerate(thresholds):
                if value <= t:
                    return i
            return len(thresholds)

        gm        = self.game_state.get_game_map()
        all_c     = gm.get_countries()
        owned     = gm.get_owned_countries(self)

        total_armies = sum(c.get_army_size() for c in all_c) or 1
        own_armies   = sum(c.get_army_size() for c in owned)
        total_c      = len(all_c) or 1

        continent_bonus = gm.get_reward(self)

        borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
        border_pressure = (
            sum(c.get_number_of_enemy_neighbors() for c in borders)
            / len(borders)
            if borders else 0.0
        )

        return (
            _bucket(own_armies / total_armies,    [0.15, 0.30, 0.50, 0.70]),
            _bucket(len(owned) / total_c,         [0.15, 0.30, 0.50, 0.70]),
            _bucket(continent_bonus,              [3, 6, 9, 12]),
            _bucket(border_pressure,              [1.0, 2.0, 3.0, 4.0]),
            self._current_phase,                  # [FIX 1]
        )

    # ── reward ───────────────────────────────────────────────────────────────
    def _compute_reward(self) -> float:
        """
        [FIX 2] Rispetto all'originale:
          · delta_a abbassato da 0.5 a 0.1
          · aggiunto continent_gain
          · aggiunta front_penalty
        """
        gm          = self.game_state.get_game_map()
        owned       = gm.get_owned_countries(self)
        n_countries = len(owned)
        n_armies    = gm.get_owned_army_size(self)
        total_c     = len(gm.get_countries()) or 1

        delta_c   = (n_countries - self._prev_countries) * 2.0
        delta_a   = (n_armies    - self._prev_armies)    * 0.1   # era 0.5
        dominance = max((n_countries / total_c - 0.3) * 5.0, 0.0)

        # bonus continenti conquistati in questo step  [FIX 2]
        new_bonus      = gm.get_reward(self)
        continent_gain = (new_bonus - self._prev_cont_bonus) * 5.0
        self._prev_cont_bonus = new_bonus

        # penalità fronti aperti  [FIX 2]
        borders       = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
        front_penalty = -len(borders) * 0.1

        self._prev_countries = n_countries
        self._prev_armies    = n_armies

        return delta_c + delta_a + dominance + continent_gain + front_penalty

    # ── aggiornamento Q ───────────────────────────────────────────────────────
    def _update_q(self, reward: float, next_state: tuple):
        if self._prev_state is None or self._prev_macro is None:
            return
        best_next  = max(self.Q[next_state].values())
        td_target  = reward + self.gamma * best_next
        self.Q[self._prev_state][self._prev_macro] += (
            self.alpha * (td_target - self.Q[self._prev_state][self._prev_macro])
        )

    # ── transizione  [FIX 3] ─────────────────────────────────────────────────
    def _record_transition(self):
        """
        Chiamato dopo ogni singola azione (place / attack / fortify),
        non una volta sola a fine turno. Questo fa sì che il segnale Q
        arrivi subito e non aggregato su tutto il turno.
        """
        next_state = self._extract_state()
        reward     = self._compute_reward()
        self._update_q(reward, next_state)
        self._prev_state = next_state

    def action_cleanup(self):
        pass  # svuotato: l'update avviene inline  [FIX 3]

    # ── selezione macro ───────────────────────────────────────────────────────
    def _select_macro(self, candidates: list[str]) -> str:
        if random.random() < self.epsilon:
            return random.choice(candidates)
        state  = self._extract_state()
        q_vals = {m: self.Q[state][m] for m in candidates}
        return max(q_vals, key=q_vals.__getitem__)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def turn_setup(self):
        self._cluster = None
        gm = self.game_state.get_game_map()
        self._prev_countries  = len(gm.get_owned_countries(self))
        self._prev_armies     = gm.get_owned_army_size(self)
        self._prev_cont_bonus = gm.get_reward(self)   # [FIX 6]
        self._current_phase   = PHASE_PLACE            # [FIX 1]
        self._prev_state      = self._extract_state()

    # ── fasi di gioco ─────────────────────────────────────────────────────────
    def place_armies(self) -> Action | None:
        self._current_phase = PHASE_PLACE              # [FIX 1]
        macro  = self._select_macro(self.PLACE_MACROS)
        self._prev_macro = macro
        action = self._execute_place_macro(macro)
        self._record_transition()                      # [FIX 3]
        return action

    def attack(self) -> Action | None:
        self._current_phase = PHASE_ATTACK             # [FIX 1]
        macro  = self._select_macro(self.ATTACK_MACROS)
        self._prev_macro = macro
        action = self._execute_attack_macro(macro)
        self._record_transition()                      # [FIX 3]
        return action

    def fortify(self) -> Action | None:
        self._current_phase = PHASE_FORTIFY            # [FIX 1]
        macro  = self._select_macro(self.FORTIFY_MACROS)
        self._prev_macro = macro
        action = self._execute_fortify_macro(macro)
        self._record_transition()                      # [FIX 3]
        return action

    # ── esecuzione macro: place ───────────────────────────────────────────────
    def _execute_place_macro(self, macro: str) -> Action | None:
        owned = self.game_state.get_game_map().get_owned_countries(self)

        if self.troops_to_place <= 0:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        target = None

        if macro == "place_contested":
            borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
            target  = utils.get_most_contested_country(borders, self)

        elif macro == "place_weakest":
            target = utils.get_weakest_friendly_country(self.game_state, self)

        elif macro == "place_continent":
            best_cont, best_val = None, -1.0
            for cont in self.game_state.get_game_map().get_continents():
                owned_in  = [c for c in cont.get_countries() if c in owned]
                n_total   = len(cont.get_countries()) or 1
                progress  = len(owned_in) / n_total
                val       = progress * (1.0 + cont.get_reward(self) / 10.0)
                if val > best_val:
                    best_val, best_cont = val, cont
            if best_cont:
                action = self.place_to_take_continent(best_cont)
                if action:
                    return action

        if target is None:
            borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
            target  = (
                utils.get_most_contested_country(borders, self)
                or (owned[0] if owned else None)
            )

        if target is None:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        self.troops_to_place -= 1
        return PlaceArmyAction(target, 1)

    # ── esecuzione macro: attack ───────────────────────────────────────────────
    def _execute_attack_macro(self, macro: str) -> Action | None:

        if macro == "attack_pass":
            # [FIX 5] penalità diretta se c'erano attacchi disponibili
            owned = self.game_state.get_game_map().get_owned_countries(self)
            feasible = any(
                c.get_army_size() > 1 and c.get_number_of_enemy_neighbors() > 0
                for c in owned
            )
            if feasible and self._prev_state is not None:
                self.Q[self._prev_state]["attack_pass"] -= 0.5

            self.add_completed_phase(GameState.ATTACK)
            return None

        if self._cluster is None:
            self._cluster = self.game_state.get_game_map().get_owned_countries(self)

        if macro == "attack_easy":
            action = self.attack_easy_expand(self._cluster)
        elif macro == "attack_fill":
            action = self.attack_fill_out(self._cluster)
        elif macro == "attack_consolidate":
            action = self.attack_consolidate(self._cluster)
        elif macro == "attack_split":
            action = self.attack_split_up(self._cluster, attack_ratio=1.2)
        else:
            action = self.attack_easy_expand(self._cluster)

        if action:
            return action

        self.add_completed_phase(GameState.ATTACK)
        return None

    # ── esecuzione macro: fortify ─────────────────────────────────────────────
    def _execute_fortify_macro(self, macro: str) -> Action | None:
        if macro == "fortify_pass":
            self.add_completed_phase(GameState.FORTIFY)
            return None

        gm      = self.game_state.get_game_map()
        owned   = gm.get_owned_countries(self)
        borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]

        if not borders:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        target = utils.get_most_contested_country(borders, self)
        if not target:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        connected = target.get_connected_friendly_countries()
        interior  = [
            c for c in connected
            if c.get_number_of_enemy_neighbors() == 0 and c.get_army_size() > 1
        ]

        if not interior:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        source = max(interior, key=lambda c: c.get_army_size())
        n      = source.get_army_size() - 1

        if n <= 0:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        self.add_completed_phase(GameState.FORTIFY)
        return FortifyAction(source, target, n)

    # ── reward terminale ──────────────────────────────────────────────────────
    def receive_terminal_reward(self, won: bool):
        """
        Chiamare una volta a fine partita.
        Applica reward terminale, fa decadere epsilon, incrementa contatore.
        """
        reward = 50.0 if won else -50.0
        if self._prev_state is not None and self._prev_macro is not None:
            self._update_q(reward, self._extract_state())

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)  # [FIX 4]
        self._games_played += 1

        result = "WIN" if won else "loss"
        print(
            f"[RLPH+ {self.color}] partita {self._games_played} → {result} | "
            f"ε={self.epsilon:.3f} | stati Q={len(self.Q)}"
        )

    # ── persistenza ───────────────────────────────────────────────────────────
    def save(self, filepath: str):
        data = {
            "epsilon":      self.epsilon,
            "games_played": self._games_played,
            "Q": {str(k): v for k, v in self.Q.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(
            f"[RLPH+ {self.color}] salvato in {filepath} "
            f"({len(self.Q)} stati, {self._games_played} partite)"
        )

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.epsilon       = data.get("epsilon", EPSILON_START)
        self._games_played = data.get("games_played", 0)
        self.Q = defaultdict(lambda: {m: 0.0 for m in self.ALL_MACROS})
        for state_str, values in data["Q"].items():
            state = tuple(int(x) for x in state_str.strip("()").split(","))
            self.Q[state] = values
        print(
            f"[RLPH+ {self.color}] caricato da {filepath} "
            f"({len(self.Q)} stati, {self._games_played} partite)"
        )

    # ── debug ─────────────────────────────────────────────────────────────────
    def explain(self) -> str:
        lines = [
            f"RLPH+ [{self.color}] — {self._games_played} partite",
            f"epsilon={self.epsilon:.3f}  alpha={self.alpha}  gamma={self.gamma}",
            f"stati Q esplorati: {len(self.Q)}",
            "",
            "Top Q-values per fase:",
        ]
        for phase_idx, phase_name in [(0, "place"), (1, "attack"), (2, "fortify")]:
            relevant = {
                k: v for k, v in self.Q.items() if len(k) == 5 and k[4] == phase_idx
            }
            if not relevant:
                lines.append(f"  {phase_name}: nessuno stato visitato")
                continue
            # media Q per macro su tutti gli stati di questa fase
            sums:   dict[str, float] = defaultdict(float)
            counts: dict[str, int]   = defaultdict(int)
            for state_vals in relevant.values():
                for macro, val in state_vals.items():
                    sums[macro]   += val
                    counts[macro] += 1
            avgs = {m: sums[m] / counts[m] for m in sums}
            best = max(avgs, key=avgs.__getitem__)
            lines.append(
                f"  {phase_name}: macro preferita = {best} "
                f"(Q medio={avgs[best]:.3f})"
            )
        return "\n".join(lines)