from __future__ import annotations

import random
import math
from collections import defaultdict

from Risk.actions import Action, PlaceArmyAction, FortifyAction
from Risk.game_state import GameState
from Risk.players.base_player import Player
from Risk import utils


# =========================================================
# FEATURES (identiche + leggermente stabilizzate)
# =========================================================
class Features:
    N = 10

    @staticmethod
    def extract(player: "PPOPlayer") -> list[float]:
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
        alive = [p for p in players if p is not player and len(gm.get_owned_countries(p)) > 0]
        enemy_ratio = min((len(alive) - 1) / 5, 1.0) if alive else 0.0

        army_sizes = [c.get_army_size() for c in owned]
        avg_a = own_armies / len(owned)

        return [
            min(own_armies / total_armies, 1.0),
            min(len(owned) / total_c, 1.0),
            len(borders) / len(owned),
            min(bp, 1.0),
            cont_progress,
            min(cont_bonus / 12.0, 1.0),
            max(enemy_ratio, 0.0),
            min(min(army_sizes) / avg_a, 1.0) if avg_a else 0.0,
            min(max(army_sizes) / avg_a, 1.0) if avg_a else 0.0,
            len(interior) / len(owned),
        ]


# =========================================================
# PPO PLAYER
# =========================================================
class PPOPlayer(Player):

    PLACE_MACROS = ["place_contested", "place_weakest", "place_continent"]
    ATTACK_MACROS = ["attack_easy", "attack_fill", "attack_consolidate",
                     "attack_split", "attack_pass"]
    FORTIFY_MACROS = ["fortify_border", "fortify_pass"]

    ALL_MACROS = PLACE_MACROS + ATTACK_MACROS + FORTIFY_MACROS

    def __init__(self, color: str):
        super().__init__(color)

        self.rng = random.Random()

        # POLICY (macro -> weights)
        self.policy_w = {
            m: [self.rng.uniform(-0.1, 0.1) for _ in range(Features.N)]
            for m in self.ALL_MACROS
        }

        # VALUE FUNCTION
        self.value_w = [0.0] * Features.N

        # MEMORY episodio
        self.memory = []

        # PARAMETRI PPO
        self.lr = 0.02
        self.gamma = 0.95
        self.clip = 0.2

        # stato precedente
        self._prev_countries = 0
        self._prev_armies = 0

        self._cluster = None

    # =========================================================
    # UTILS
    # =========================================================
    def _dot(self, w, x):
        return sum(wi * xi for wi, xi in zip(w, x))

    def _softmax(self, logits):
        m = max(logits.values())
        exps = {k: math.exp(v - m) for k, v in logits.items()}
        s = sum(exps.values())
        return {k: v / s for k, v in exps.items()}

    def _policy(self, features, candidates):
        logits = {
            m: self._dot(self.policy_w[m], features)
            for m in candidates
        }
        return self._softmax(logits)

    def _sample(self, probs):
        r = self.rng.random()
        acc = 0
        for k, v in probs.items():
            acc += v
            if r <= acc:
                return k
        return list(probs.keys())[0]

    def _value(self, features):
        return self._dot(self.value_w, features)

    # =========================================================
    # CHOOSE ACTION
    # =========================================================
    def _choose_macro(self, candidates):
        features = Features.extract(self)
        probs = self._policy(features, candidates)
        action = self._sample(probs)

        self.memory.append({
            "features": features,
            "action": action,
            "probs": probs
        })

        return action

    # =========================================================
    # PPO UPDATE
    # =========================================================
    def _update(self, rewards):

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        for step, G in zip(self.memory, returns):
            f = step["features"]
            a = step["action"]
            old_prob = step["probs"].get(a, 1e-8)

            V = self._value(f)
            advantage = G - V

            # update VALUE
            for i in range(Features.N):
                self.value_w[i] += self.lr * advantage * f[i]

            # nuova policy
            new_probs = self._policy(f, self.ALL_MACROS)
            new_prob = new_probs.get(a, 1e-8)

            ratio = new_prob / (old_prob + 1e-8)
            clipped = max(min(ratio, 1 + self.clip), 1 - self.clip)
            grad = min(ratio * advantage, clipped * advantage)

            for i in range(Features.N):
                self.policy_w[a][i] += self.lr * grad * f[i]

        self.memory.clear()

    # =========================================================
    # REWARD (FIXED)
    # =========================================================
    def _compute_reward(self):
        gm = self.game_state.get_game_map()
        owned = gm.get_owned_countries(self)

        n_c = len(owned)
        n_a = gm.get_owned_army_size(self)

        delta_c = (n_c - self._prev_countries) * 3.0
        delta_a = (n_a - self._prev_armies) * 0.2

        self._prev_countries = n_c
        self._prev_armies = n_a

        return delta_c + delta_a

    # =========================================================
    # GAME FLOW
    # =========================================================
    def turn_setup(self):
        gm = self.game_state.get_game_map()
        self._prev_countries = len(gm.get_owned_countries(self))
        self._prev_armies = gm.get_owned_army_size(self)
        self._cluster = None

    def action_cleanup(self):
        reward = self._compute_reward()
        self._update([reward])

    # =========================================================
    # PLACE
    # =========================================================
    def place_armies(self) -> Action | None:
        macro = self._choose_macro(self.PLACE_MACROS)
        owned = self.game_state.get_game_map().get_owned_countries(self)

        if self.troops_to_place <= 0 or not owned:
            self.add_completed_phase(GameState.PLACE_ARMY)
            return None

        if macro == "place_weakest":
            target = utils.get_weakest_friendly_country(self.game_state, self)

        elif macro == "place_contested":
            borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]
            target = utils.get_most_contested_country(borders, self)

        else:
            target = owned[0]

        if target is None:
            target = owned[0]

        self.troops_to_place -= 1
        return PlaceArmyAction(target, 1)

    # =========================================================
    # ATTACK
    # =========================================================
    def attack(self) -> Action | None:
        macro = self._choose_macro(self.ATTACK_MACROS)

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
            action = None

        if action:
            return action

        self.add_completed_phase(GameState.ATTACK)
        return None

    # =========================================================
    # FORTIFY
    # =========================================================
    def fortify(self) -> Action | None:
        macro = self._choose_macro(self.FORTIFY_MACROS)

        if macro == "fortify_pass":
            self.add_completed_phase(GameState.FORTIFY)
            return None

        gm = self.game_state.get_game_map()
        owned = gm.get_owned_countries(self)

        borders = [c for c in owned if c.get_number_of_enemy_neighbors() > 0]

        if not borders:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        target = utils.get_most_contested_country(borders, self)
        if not target:
            self.add_completed_phase(GameState.FORTIFY)
            return None

        interior = [
            c for c in target.get_connected_friendly_countries()
            if c.get_number_of_enemy_neighbors() == 0 and c.get_army_size() > 1
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

    # =========================================================
    # END GAME
    # =========================================================
    def receive_terminal_reward(self, won: bool):
        reward = 50.0 if won else -25.0
        self._update([reward])