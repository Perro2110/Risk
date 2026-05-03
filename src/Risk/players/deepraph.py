from __future__ import annotations

import random
import json
from collections import defaultdict

from Risk.actions import Action, PlaceArmyAction, FortifyAction
from Risk.game_state import GameState
from Risk.map import Country
from Risk import utils
from Risk.players.smart_player import SmartPlayer


# ---------------------------------------------------------------------------
# DeepRLPH - Risk Q-learning, terza versione
# ---------------------------------------------------------------------------
# Fix rispetto a RLPHPlus:
#
#   FIX A - macro scelta UNA VOLTA per fase, non per ogni tick.
#            Il Q-update avviene una sola volta quando la fase finisce,
#            non centinaia di volte per turno con delta quasi-zero.
#            Questo è il fix più importante: senza di esso il segnale
#            è puro rumore.
#
#   FIX B - reward calcolata a fine fase su delta accumulato dall'inizio
#            della fase, non dal tick precedente. I delta non si azzerano
#            ogni tick, quindi i segnali netti (conquista paese, continente)
#            non vengono diluiti.
#
#   FIX C - reward shaping aggressivo e gerarchico:
#            · eliminazione giocatore    +100
#            · conquista continente      +30 per punto bonus
#            · conquista paese           +5
#            · perdita paese             -8  (asimmetria intenzionale)
#            · perdita continente        -20 per punto bonus
#            · delta armate             ±0.05 (quasi silenzio)
#            I reward intermedi sono secondari; il terminale (+100/-100)
#            domina e propaga all'indietro via gamma.
#
#   FIX D - attack_pass non tocca più Q direttamente. Viene trattato
#            come qualsiasi altro macro: il reward naturale della fase
#            (spesso negativo se i nemici avanzano) penalizza da solo
#            la passività.
#
#   FIX E - stato arricchito con feature tattiche:
#            · continent_progress: quanto siamo vicini a completare
#              il continente più promettente (0-4 bucket)
#            · can_attack: abbiamo almeno un attacco disponibile (0/1)
#            Rimuovere border_pressure (era media globale inutile).
#
#   FIX F - action_cleanup rimosso. Ogni aggiornamento Q è esplicito
#            e avviene una sola volta per fase in _end_phase().
# ---------------------------------------------------------------------------


#  iperparametri
ALPHA = 0.15   # learning rate leggermente più alto per convergenza più veloce
GAMMA = 0.95   # discount più alto: le conseguenze future contano di più
EPSILON_START = 0.80
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995  # moltiplicatore per partita

# reward shaping
R_ELIMINATE_PLAYER = 100.0
R_COUNTRY_GAIN = 5.0
R_COUNTRY_LOSS = -8.0   # perdere è peggio che guadagnare (asimmetria)
R_CONTINENT_GAIN = 30.0   # per punto bonus del continente
R_CONTINENT_LOSS = -20.0
R_ARMY_SCALE = 0.05  # quasi silenzio - solo rumore di fondo
R_WIN = 100.0
R_LOSE = -100.0

# indici di fase
PHASE_PLACE = 0
PHASE_ATTACK = 1
PHASE_FORTIFY = 2


class DeepRLPH(SmartPlayer):
    """
    Q-learning tabulare con macro-azioni, versione corretta.

    Cambiamento architetturale principale: il macro viene scelto una sola
    volta per fase e mantenuto per tutti i tick di quella fase. Il Q-update
    avviene una sola volta quando la fase termina, calcolando il reward
    sull'intero delta accumulato durante la fase.

    Args:
        color           colore del player
        alpha           learning rate           (default 0.15)
        gamma           discount factor         (default 0.95)
        epsilon         esplorazione iniziale   (default 0.80)
        troops_to_place truppe iniziali
    """

    def __init__(
        self,
        color: str,
        alpha:   float = ALPHA,
        gamma:   float = GAMMA,
        epsilon: float = EPSILON_START,
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

        # stato all'inizio della fase corrente
        self._phase_start_state:    tuple | None = None
        self._phase_start_macro:      str | None = None

        # snapshot metriche all'inizio della fase - per delta reward
        self._phase_start_countries:   int = 0
        self._phase_start_armies:      int = 0
        self._phase_start_cont_bonus:  float = 0.0
        self._phase_start_n_players:   int = 0  # per rilevare eliminazioni

        # macro bloccata per la fase corrente [FIX A]
        self._current_phase_macro: str | None = None
        self._current_phase:       int = PHASE_PLACE

        self._cluster: list[Country] | None = None
        self._games_played: int = 0

    def _extract_state(self) -> tuple:
        """
        Tupla discreta che rappresenta la situazione corrente.

        Feature:
            0  army_ratio         armate proprie / totale  (5 bucket)
            1  country_ratio      paesi propri / totale    (5 bucket)
            2  continent_bonus    bonus continenti totali  (5 bucket)
            3  continent_progress quanto siamo vicini a completare
                                  il continente più promettente (5 bucket)
            4  can_attack         1 se abbiamo almeno un attacco possibile
            5  phase              0=place  1=attack  2=fortify
        """
        def _bucket(value: float, thresholds: list[float]) -> int:
            for i, t in enumerate(thresholds):
                if value <= t:
                    return i
            return len(thresholds)

        gm = self.game_state.get_game_map()
        all_c = gm.get_countries()
        owned = gm.get_owned_countries(self)

        total_armies = sum(c.get_army_size() for c in all_c) or 1
        own_armies = sum(c.get_army_size() for c in owned)
        total_c = len(all_c) or 1

        continent_bonus = gm.get_reward(self)

        # continent_progress: (paesi mancanti al continente più vicino)
        # bucket: 0 = già completo, 4 = mancano 4+ paesi
        best_progress = 4  # pessimistic default
        for cont in gm.get_continents():
            cont_countries = cont.get_countries()
            owned_in = sum(1 for c in cont_countries if c in owned)
            missing = len(cont_countries) - owned_in
            if missing < best_progress:
                best_progress = missing
        continent_progress = min(best_progress, 4)

        # can_attack: 1 se almeno un paese può attaccare
        can_attack = int(any(
            c.get_army_size() > 1 and c.get_number_of_enemy_neighbors() > 0
            for c in owned
        ))

        return (
            _bucket(own_armies / total_armies, [0.10, 0.25, 0.45, 0.65]),
            _bucket(len(owned) / total_c,      [0.10, 0.25, 0.45, 0.65]),
            _bucket(continent_bonus,           [2, 5, 9, 14]),
            continent_progress,
            can_attack,
            self._current_phase,
        )

    def _compute_phase_reward(self) -> float:
        """
        Reward calcolata UNA VOLTA alla fine della fase, sul delta accumulato
        dall'inizio della fase stessa.

        Non viene chiamata ogni tick: il delta non si azzera mai a metà.
        """
        gm = self.game_state.get_game_map()
        owned = gm.get_owned_countries(self)

        n_countries = len(owned)
        n_armies = gm.get_owned_army_size(self)
        cont_bonus = gm.get_reward(self)
        n_players_now = len(self.game_state.get_players())

        # delta rispetto all'inizio della fase
        delta_countries = n_countries - self._phase_start_countries
        delta_armies = n_armies - self._phase_start_armies
        delta_bonus = cont_bonus - self._phase_start_cont_bonus

        # eliminazione avversari - segnale fortissimo
        eliminations = max(0, self._phase_start_n_players - n_players_now)

        reward = 0.0

        # conquista / perdita paesi  [FIX C]
        if delta_countries > 0:
            reward += delta_countries * R_COUNTRY_GAIN
        else:
            reward += delta_countries * abs(R_COUNTRY_LOSS)  # già negativo

        # conquista / perdita continenti  [FIX C]
        if delta_bonus > 0:
            reward += delta_bonus * R_CONTINENT_GAIN
        elif delta_bonus < 0:
            reward += delta_bonus * abs(R_CONTINENT_LOSS)

        # eliminazione  [FIX C]
        reward += eliminations * R_ELIMINATE_PLAYER

        # armate - quasi silenzio, evita che il segnale venga dominato
        # da variazioni normali durante il combattimento  [FIX C]
        reward += delta_armies * R_ARMY_SCALE

        return reward

    #  aggiornamento Q
    def _update_q(self, state: tuple, macro: str, reward: float,
                  next_state: tuple):
        best_next = max(self.Q[next_state].values())
        td_target = reward + self.gamma * best_next
        self.Q[state][macro] += self.alpha * (
            td_target - self.Q[state][macro]
        )

    def _end_phase(self, phase_const: int):
        """
        Chiamato UNA SOLA VOLTA quando una fase termina.
        Calcola il reward sull'intero delta di fase e aggiorna Q.
        """
        if self._phase_start_state is None or self._phase_start_macro is None:
            self.add_completed_phase(phase_const)
            return

        reward = self._compute_phase_reward()
        next_state = self._extract_state()
        self._update_q(
            self._phase_start_state,
            self._phase_start_macro,
            reward,
            next_state,
        )

        # resetta per la fase successiva
        self._current_phase_macro = None
        self._phase_start_state = None
        self._phase_start_macro = None
        self.add_completed_phase(phase_const)

    #  snapshot inizio fase
    def _begin_phase(self, phase: int):
        """
        Chiamato al primo tick di una nuova fase.
        Scatta lo snapshot delle metriche e sceglie il macro per tutta la fase.
        """
        self._current_phase = phase

        gm = self.game_state.get_game_map()
        self._phase_start_countries = len(gm.get_owned_countries(self))
        self._phase_start_armies = gm.get_owned_army_size(self)
        self._phase_start_cont_bonus = gm.get_reward(self)
        self._phase_start_n_players = len(self.game_state.get_players())
        self._phase_start_state = self._extract_state()

        # sceglie il macro UNA VOLTA per tutta la fase
        if phase == PHASE_PLACE:
            candidates = self.PLACE_MACROS
        elif phase == PHASE_ATTACK:
            candidates = self.ATTACK_MACROS
        else:
            candidates = self.FORTIFY_MACROS

        self._current_phase_macro = self._select_macro(candidates)
        self._phase_start_macro = self._current_phase_macro

    #  selezione macro
    def _select_macro(self, candidates: list[str]) -> str:
        if random.random() < self.epsilon:
            return random.choice(candidates)
        state = self._extract_state()
        q_vals = {m: self.Q[state][m] for m in candidates}
        return max(q_vals, key=q_vals.__getitem__)

    #  lifecycle
    def turn_setup(self):
        self._cluster = None
        self._current_phase_macro = None
        self._phase_start_state = None
        self._phase_start_macro = None
        self._current_phase = PHASE_PLACE

    def action_cleanup(self):
        pass  # tutti gli update avvengono in _end_phase()

    def place_armies(self) -> Action | None:
        # primo tick della fase: inizializza
        if self._current_phase_macro is None:
            self._begin_phase(PHASE_PLACE)

        action = self._execute_place_macro(self._current_phase_macro)

        if action is None:
            # la fase è finita - aggiorna Q una sola volta
            self._end_phase(GameState.PLACE_ARMY)

        return action

    def attack(self) -> Action | None:
        if self._current_phase_macro is None:
            self._begin_phase(PHASE_ATTACK)

        action = self._execute_attack_macro(self._current_phase_macro)

        if action is None:
            self._end_phase(GameState.ATTACK)

        return action

    def fortify(self) -> Action | None:
        if self._current_phase_macro is None:
            self._begin_phase(PHASE_FORTIFY)

        action = self._execute_fortify_macro(self._current_phase_macro)

        if action is None:
            self._end_phase(GameState.FORTIFY)

        return action

    def receive_terminal_reward(self, won: bool):
        """
        Chiamare una volta a fine partita. Il reward terminale (+100/-100)
        è molto più grande dei reward intermedi,
        quindi propaga significativamente all'indietro via gamma.
        """
        reward = R_WIN if won else R_LOSE

        if self._phase_start_state is not None \
                and self._phase_start_macro is not None:
            next_state = self._extract_state()
            self._update_q(
                self._phase_start_state,
                self._phase_start_macro,
                reward,
                next_state,
            )

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self._games_played += 1

    def save(self, filepath: str):
        data = {
            "epsilon":      self.epsilon,
            "games_played": self._games_played,
            "Q": {str(k): v for k, v in self.Q.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(
            f"[RLPHPlusPlus {self.color}] salvato in {filepath} "
            f"({len(self.Q)} stati, {self._games_played} partite)"
        )

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.epsilon = data.get("epsilon", EPSILON_START)
        self._games_played = data.get("games_played", 0)
        self.Q = defaultdict(lambda: {m: 0.0 for m in self.ALL_MACROS})
        for state_str, values in data["Q"].items():
            state = tuple(int(x) for x in state_str.strip("()").split(","))
            self.Q[state] = values
        print(
            f"[RLPHPlusPlus {self.color}] caricato da {filepath} "
            f"({len(self.Q)} stati, {self._games_played} partite)"
        )

    def explain(self) -> str:
        lines = [
            f"RLPHPlusPlus [{self.color}] - {self._games_played} partite",
            f"epsilon={self.epsilon:.4f}"
            f"alpha={self.alpha} gamma={self.gamma}",
            f"stati Q esplorati: {len(self.Q)}",
            "",
            "Top Q-values medi per fase:",
        ]
        # stato ha 6 elementi, fase in posizione 5
        phases = [(0, "place"), (1, "attack"), (2, "fortify")]
        for phase_idx, phase_name in phases:
            relevant = {
                k: v for k, v in self.Q.items()
                if len(k) == 6 and k[5] == phase_idx
            }
            if not relevant:
                lines.append(f"  {phase_name}: nessuno stato visitato")
                continue
            sums:   dict[str, float] = defaultdict(float)
            counts: dict[str, int] = defaultdict(int)
            for state_vals in relevant.values():
                for macro, val in state_vals.items():
                    sums[macro] += val
                    counts[macro] += 1
            avgs = {m: sums[m] / counts[m] for m in sums}
            sorted_macros = sorted(avgs.items(), key=lambda kv: -kv[1])
            lines.append(f"  {phase_name}:")
            for macro, val in sorted_macros:
                bar = "█" * max(0, int((val / 10) + 5))
                lines.append(f"    {macro:<25} Q={val:+8.2f}  {bar}")
        return "\n".join(lines)
