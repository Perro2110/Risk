from __future__ import annotations

import random
import json
import math
from collections import defaultdict

from Risk.actions import Action, PlaceArmyAction, FortifyAction
from Risk.game_state import GameState
from Risk.map import Country
from Risk import utils
from Risk.players.smart_player import SmartPlayer


# ---------------------------------------------------------------------------
# Menny
# ---------------------------------------------------------------------------
# Architettura: ensemble di N alberi decisionali binari (depth fissa).
# Ogni albero è costruito su feature estratte dallo stato di gioco e
# produce un voto su quale macro-azione eseguire.
#
# Il sistema di pesatura è ispirato a Gradient Boosting (XGBoost-style):
#   - ogni albero ha un peso w_i inizializzato a 1/N
#   - dopo ogni azione: si calcola un "errore" (reward negativo = errore alto)
#   - si applica un update moltiplicativo ai pesi degli alberi che hanno
#     votato la macro eseguita:
#       w_i *= exp(learning_rate * reward)   se ha votato la macro scelta
#       w_i *= exp(-learning_rate * |reward| * penalty_factor)  altrimenti
#   - i pesi vengono normalizzati dopo ogni update
#   - ogni tot partite gli alberi con peso < soglia vengono rigenerati
#     (pruning + regrowth, come nel gradient boosting si aggiunge un nuovo
#     weak learner a correggere l'errore residuo)
# ---------------------------------------------------------------------------

#  costanti globali
N_TREES = 100        # numero di alberi nell'ensemble
TREE_DEPTH = 4         # profondità massima di ogni albero
LEARNING_RATE = 0.15      # quanto velocemente aggiornare i pesi
PENALTY_FACTOR = 0.5       # penalità relativa per chi NON ha votato la scelta
PRUNE_THRESH = 0.01      # peso minimo prima del pruning
PRUNE_EVERY = 30        # rigenera alberi deboli ogni N partite
EPSILON_START = 0.25      # esplorazione iniziale
EPSILON_MIN = 0.04      # esplorazione minima
EPSILON_DECAY = 0.992     # moltiplicatore per partita


#  feature extractor
class Features:
    """
    Calcola un vettore numerico normalizzato [0,1] dalla situazione di gioco.
    Ogni feature è descritta qui per rendere gli alberi interpretabili.

    Indici:
        0  army_ratio          armate proprie / totale armate
        1  country_ratio       paesi propri / totale paesi
        2  border_ratio        paesi di confine / paesi propri
        3  avg_border_pressure media nemici per paese di confine (max 6) /6
        4  continent_progress  media % completamento di ogni continente
        5  continent_bonus     rinforzi bonus da continenti (normaliz. su 12)
        6  enemy_count_ratio   (n_nemici vivi - 1) / 5  (clip 0-1)
        7  weakest_ratio       armate paese più debole / media armate proprie
        8  strongest_ratio     armate paese più forte / media armate proprie
        9  interior_ratio      paesi interni (no nemici vicini) / paesi propri
    """
    N = 10

    @staticmethod
    def extract(player: "MENNY") -> list[float]:
        gs = player.game_state
        gm = gs.get_game_map()
        all_c = gm.get_countries()
        owned = gm.get_owned_countries(player)

        if not owned:
            return [0.0] * Features.N

        total_armies = sum(c.get_army_size() for c in all_c) or 1
        own_armies = sum(c.get_army_size() for c in owned) or 1
        total_c = len(all_c) or 1

        borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
        interior = [c for c in owned if c.get_number_of_enemy_neighbors() == 0]

        bp = (sum(c.get_number_of_enemy_neighbors() for c in borders) /
              (len(borders) * 6)) if borders else 0.0

        cont_prog_vals = []
        for cont in gm.get_continents():
            cont_c = cont.get_countries()
            owned_in = [c for c in cont_c if c in owned]
            cont_prog_vals.append(len(owned_in) / (len(cont_c) or 1))
        cont_progress = sum(cont_prog_vals) / (len(cont_prog_vals) or 1)

        cont_bonus = gm.get_reward(player)

        players = gs.get_players()
        alive = [p for p in players
                 if p is not player and len(gm.get_owned_countries(p)) > 0]
        enemy_ratio = min((len(alive) - 1) / 5, 1.0) if alive else 0.0

        army_sizes = [c.get_army_size() for c in owned]
        avg_a = own_armies / len(owned)
        weakest_r = min(army_sizes) / avg_a if avg_a else 0.0
        strongest_r = min(max(army_sizes) / avg_a, 3.0) / 3.0 if avg_a else 0.0

        return [
            min(own_armies / total_armies, 1.0),                  # 0
            min(len(owned) / total_c,     1.0),                   # 1
            len(borders) / len(owned),                            # 2
            min(bp, 1.0),                                         # 3
            cont_progress,                                        # 4
            min(cont_bonus / 12.0, 1.0),                          # 5
            max(enemy_ratio, 0.0),                                # 6
            min(weakest_r,  2.0) / 2.0,                           # 7
            strongest_r,                                          # 8
            len(interior) / len(owned),                           # 9
        ]


#  nodo dell'albero
class DTNode:
    """
    Nodo di un albero decisionale binario.
    Se foglia, contiene direttamente il label (macro-azione).
    Se interno, contiene feature_index e threshold.
    """
    __slots__ = ("feature_idx", "threshold", "left", "right", "label")

    def __init__(
        self,
        feature_idx: int | None = None,
        threshold: float = 0.5,
        left: "DTNode | None" = None,
        right: "DTNode | None" = None,
        label: str | None = None,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label   # None se nodo interno

    @property
    def is_leaf(self) -> bool:
        return self.label is not None

    def predict(self, features: list[float]) -> str | None:
        node = self
        while node is not None and not node.is_leaf:
            if node.feature_idx is None:
                break

            if features[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label if node else None

    #  serializzazione
    def to_dict(self) -> dict:
        if self.is_leaf:
            return {"label": self.label}
        return {
            "fi": self.feature_idx,
            "th": self.threshold,
            "l":  self.left.to_dict() if self.left else None,
            "r":  self.right.to_dict() if self.right else None,
        }

    @staticmethod
    def from_dict(d: dict) -> "DTNode":
        if "label" in d:
            return DTNode(label=d["label"])
        return DTNode(
            feature_idx=d["fi"],
            threshold=d["th"],
            left=DTNode.from_dict(d["l"]),
            right=DTNode.from_dict(d["r"]),
        )


#  costruzione casuale di un albero
def _build_random_tree(
    macros: list[str],
    depth: int,
    rng: random.Random,
) -> DTNode:
    """
    Costruisce un albero binario casuale di profondità `depth`.
    Ogni nodo interno sceglie una feature e una soglia a caso.
    Le foglie sono assegnate casualmente tra le macro disponibili.
    L'albero viene costruito in modo da avere distribuzione ragionevolmente
    uniforme delle foglie (ogni foglia prende una macro diversa ciclicamente).
    """
    if depth == 0:
        return DTNode(label=rng.choice(macros))

    fi = rng.randint(0, Features.N - 1)
    thr = round(rng.uniform(0.2, 0.8), 2)
    return DTNode(
        feature_idx=fi,
        threshold=thr,
        left=_build_random_tree(macros, depth - 1, rng),
        right=_build_random_tree(macros, depth - 1, rng),
    )


def _build_biased_tree(
    macros: list[str],
    depth: int,
    rng: random.Random,
    bias: dict[str, float],          # macro -> peso preferenziale
) -> DTNode:
    """
    Come _build_random_tree ma le foglie vengono assegnate con probabilità
    proporzionale al bias dizionario. Usato per rigenerare alberi "corretti"
    verso le azioni che storicamente hanno portato reward positivo.
    """
    if depth == 0:
        total = sum(bias.values()) or 1
        r = rng.random() * total
        acc = 0.0
        for m, w in bias.items():
            acc += w
            if r <= acc:
                return DTNode(label=m)
        return DTNode(label=rng.choice(macros))

    fi = rng.randint(0, Features.N - 1)
    thr = round(rng.uniform(0.2, 0.8), 2)
    return DTNode(
        feature_idx=fi,
        threshold=thr,
        left=_build_biased_tree(macros, depth - 1, rng, bias),
        right=_build_biased_tree(macros, depth - 1, rng, bias),
    )


#  il player
class MENNY(SmartPlayer):
    """
    Multiple ENsemble Neural-like strategY.

    Ensemble di alberi decisionali con pesatura online stile XGBoost:
    ogni albero vota una macro-azione, la decisione finale è la macro
    con il punteggio pesato più alto. I pesi si aggiornano moltiplicativamente
    in base alla reward osservata dopo ogni azione.

    Args:
        color           colore del player
        n_trees         numero di alberi nell'ensemble  (default 12)
        depth           profondità di ogni albero        (default 4)
        learning_rate   velocità aggiornamento pesi      (default 0.15)
        epsilon         esplorazione iniziale            (default 0.25)
        troops_to_place truppe iniziali (come Player)
    """

    def __init__(
        self,
        color: str,
        n_trees: int = N_TREES,
        depth: int = TREE_DEPTH,
        learning_rate: float = LEARNING_RATE,
        epsilon: float = EPSILON_START,
        troops_to_place: int = 0,
    ):
        super().__init__(color, troops_to_place)
        self._rng = random.Random()  # RNG interno isolato
        self.lr = learning_rate
        self.epsilon = epsilon
        self.n_trees = n_trees
        self.depth = depth

        # ensemble: lista di (DTNode, weight)
        self._trees: list[DTNode] = [
            _build_random_tree(self.ALL_MACROS, depth, self._rng)
            for _ in range(n_trees)
        ]
        self._weights: list[float] = [1.0 / n_trees] * n_trees

        # bias accumulato per la rigenerazione (macro -> reward_sum)
        self._macro_reward_acc: dict[str, float] = {
            m: 1.0 for m in self.ALL_MACROS
        }

        # memoria di transizione
        self._prev_features:   list[float] | None = None
        self._prev_macro:      str | None = None
        self._prev_countries:  int = 0
        self._prev_armies:     int = 0
        self._prev_cont_bonus: float = 0.0

        self._cluster: list[Country] | None = None
        self._games_played: int = 0

    #  ensemble predict
    def _ensemble_vote(
        self,
        features: list[float],
        candidates: list[str],
    ) -> str:
        """
        Raccoglie i voti pesati degli alberi per le macro candidate.
        Ritorna la macro con punteggio pesato più alto.
        """
        scores: dict[str, float] = defaultdict(float)
        for tree, w in zip(self._trees, self._weights):
            vote = tree.predict(features)
            if vote in candidates:
                scores[vote] += w
        if not scores:
            return self._rng.choice(candidates)
        return max(scores, key=scores.__getitem__)

    def _select_macro(self, candidates: list[str]) -> str:
        """Epsilon-greedy sulla votazione ensemble."""
        if self._rng.random() < self.epsilon:
            return self._rng.choice(candidates)
        features = Features.extract(self)
        return self._ensemble_vote(features, candidates)

    #  aggiornamento pesi (core boosting online)
    def _update_weights(self, macro: str, reward: float):
        """
        Update moltiplicativo stile AdaBoost/XGBoost online:

          voted the chosen macro → multiply by exp(+lr * reward)
          did NOT vote it        → multiply by exp(-lr * |reward| * penalty)

        Poi normalizza i pesi a somma 1.
        Accumula il reward per la rigenerazione bias.
        """
        self._macro_reward_acc[macro] = (
            self._macro_reward_acc.get(macro, 0.0) + reward
        )

        features = self._prev_features or [0.0] * Features.N
        new_w = []
        for tree, w in zip(self._trees, self._weights):
            tree_vote = tree.predict(features)
            if tree_vote == macro:
                # albero corretto: rinforza
                factor = math.exp(self.lr * reward)
            else:
                # albero scorretto: penalizza proporzionalmente
                factor = math.exp(-self.lr * abs(reward) * PENALTY_FACTOR)
            new_w.append(w * factor)

        # normalizza
        total = sum(new_w) or 1.0
        self._weights = [w / total for w in new_w]

    def _normalize_weights(self):
        total = sum(self._weights) or 1.0
        self._weights = [w / total for w in self._weights]

    #  pruning e rigenerazione alberi deboli
    def _maybe_prune_and_regrow(self):
        """
        Ogni PRUNE_EVERY partite, gli alberi con peso < PRUNE_THRESH vengono
        sostituiti con alberi biased verso le macro con reward_acc più alta.
        Simula l'aggiunta di weak learner correttivi nel gradient boosting.
        """
        if self._games_played % PRUNE_EVERY != 0:
            return

        # costruisci il bias normalizzato (solo valori positivi)
        pos_acc = {
            m: max(v, 0.0)
            for m, v in self._macro_reward_acc.items()
        }
        total_pos = sum(pos_acc.values()) or 1.0
        bias = {m: v / total_pos for m, v in pos_acc.items()}

        n_replaced = 0
        for i, (tree, w) in enumerate(zip(self._trees, self._weights)):
            if w < PRUNE_THRESH:
                self._trees[i] = _build_biased_tree(
                    self.ALL_MACROS, self.depth, self._rng, bias
                )
                self._weights[i] = 1.0 / self.n_trees
                n_replaced += 1

        if n_replaced:
            self._normalize_weights()
            print(
                f"[MENNY {self.color}] partita {self._games_played}: "
                f"rigenerati {n_replaced} alberi deboli"
            )

    #  reward
    def _compute_reward(self) -> float:
        gm = self.game_state.get_game_map()
        owned = gm.get_owned_countries(self)
        n_countries = len(owned)
        n_armies = gm.get_owned_army_size(self)
        total_c = len(gm.get_countries()) or 1

        delta_c = (n_countries - self._prev_countries) * 2.0
        delta_a = (n_armies - self._prev_armies) * 0.3
        dominance = max((n_countries / total_c - 0.3) * 4.0, 0.0)

        new_bonus = gm.get_reward(self)
        cont_gain = (new_bonus - self._prev_cont_bonus) * 5.0
        self._prev_cont_bonus = new_bonus

        borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
        front_penalty = len(borders) * 0.1

        self._prev_countries = n_countries
        self._prev_armies = n_armies

        return delta_c + delta_a + dominance + cont_gain + front_penalty

    #  transizione
    def _record_transition(self):
        """Calcola reward e aggiorna i pesi dell'ensemble."""
        reward = self._compute_reward()
        features = Features.extract(self)
        if self._prev_macro is not None:
            self._update_weights(self._prev_macro, reward)
        self._prev_features = features

    #  lifecycle
    def turn_setup(self):
        super().turn_setup()
        gm = self.game_state.get_game_map()
        self._prev_countries = len(gm.get_owned_countries(self))
        self._prev_armies = gm.get_owned_army_size(self)
        self._prev_cont_bonus = gm.get_reward(self)
        self._prev_features = Features.extract(self)

    def action_cleanup(self):
        pass   # update avviene inline dopo ogni azione

    #  fasi di gioco
    def place_armies(self) -> Action | None:
        macro = self._select_macro(self.PLACE_MACROS)
        self._prev_macro = macro
        action = self._execute_place_macro(macro)
        self._record_transition()
        return action

    def attack(self) -> Action | None:
        macro = self._select_macro(self.ATTACK_MACROS)
        self._prev_macro = macro
        action = self._execute_attack_macro(macro)
        self._record_transition()
        return action

    def fortify(self) -> Action | None:
        macro = self._select_macro(self.FORTIFY_MACROS)
        self._prev_macro = macro
        action = self._execute_fortify_macro(macro)
        self._record_transition()
        return action

    #  reward terminale
    def receive_terminal_reward(self, won: bool):
        """
        Chiamare una volta a fine partita.
        Applica un reward terminale grande, aggiorna epsilon
        e gestisce il pruning.
        """
        reward = 50.0 if won else -25.0
        if self._prev_macro is not None:
            self._update_weights(self._prev_macro, reward)

        # epsilon decay
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        self._games_played += 1
        self._maybe_prune_and_regrow()

        # result = "WIN" if won else "loss"
        # print(
        #     f"[MENNY {self.color}] partita {self._games_played} → {result} | "
        #     f"ε={self.epsilon:.3f} | "
        #     f"w_range=[{min(self._weights):.3f}, {max(self._weights):.3f}]"
        # )

    #  persistenza
    def save(self, filepath: str):
        """Salva ensemble, pesi e statistiche su JSON."""
        data = {
            "n_trees":          self.n_trees,
            "depth":            self.depth,
            "learning_rate":    self.lr,
            "epsilon":          self.epsilon,
            "games_played":     self._games_played,
            "weights":          self._weights,
            "macro_reward_acc": self._macro_reward_acc,
            "trees":            [t.to_dict() for t in self._trees],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # print(
        #     f"[MENNY {self.color}] salvato in {filepath} "
        #     f"({self.n_trees} alberi, {self._games_played} partite)"
        # )

    def load(self, filepath: str):
        """Carica un ensemble salvato."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.n_trees = data["n_trees"]
        self.depth = data["depth"]
        self.lr = data["learning_rate"]
        self.epsilon = data["epsilon"]
        self._games_played = data["games_played"]
        self._weights = data["weights"]
        self._macro_reward_acc = data["macro_reward_acc"]
        self._trees = [DTNode.from_dict(d) for d in data["trees"]]
        # print(
        #     f"[MENNY {self.color}] caricato da {filepath} "
        #     f"({self.n_trees} alberi, {self._games_played} partite)"
        # )

    #  debug / analisi
    def explain(self) -> str:
        """
        Ritorna una stringa leggibile che mostra il peso attuale di ogni albero
        e le macro più votate complessivamente.
        """
        lines = [f"MENNY [{self.color}] {self._games_played} partite giocate"]
        lines.append(f"epsilon = {self.epsilon:.3f}   lr = {self.lr}")
        lines.append("")
        lines.append("Pesi alberi:")
        for i, (tree, w) in enumerate(zip(self._trees, self._weights)):
            lines.append(f"  albero {i:2d}: peso={w:.4f}")
        lines.append("")
        lines.append("Reward accumulato per macro:")
        for m, v in sorted(
            self._macro_reward_acc.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {m:<25s}: {v:+.2f}")
        return "\n".join(lines)
