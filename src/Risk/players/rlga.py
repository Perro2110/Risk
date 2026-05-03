from __future__ import annotations

import random
import json
import copy
from typing import Optional

from Risk.actions import Action, PlaceArmyAction, FortifyAction
from Risk.game_state import GameState
from Risk.map import Country
from Risk import utils
from Risk.players.smart_player import SmartPlayer


# ---------------------------------------------------------------------------
# RLGA - Risk Learning con Algoritmo Genetico
# ---------------------------------------------------------------------------
# Invece di Q-learning, ogni agente ha un GENOMA:
#   un dizionario  {fase: {macro: peso_float}}
#
# La selezione della macro avviene con softmax sui pesi → comportamento
# stocastico ma guidato dall'evoluzione.
#
# Il training (train_rlga.py) fa evolvere una POPOLAZIONE di agenti:
#   1. valuta fitness (win-rate su K partite)
#   2. selezione torneo
#   3. crossover uniforme
#   4. mutazione gaussiana
#   5. élitismo: il best sopravvive sempre
# ---------------------------------------------------------------------------

def _softmax_choice(
            weights: dict[str, float], temperature: float = 1.0
        ) -> str:
    """Selezione stocastica con softmax sui pesi."""
    import math
    keys = list(weights.keys())
    vals = [weights[k] / temperature for k in keys]
    max_v = max(vals)
    exps = [math.exp(v - max_v) for v in vals]
    total = sum(exps)
    probs = [e / total for e in exps]
    r = random.random()
    cumulative = 0.0
    for key, prob in zip(keys, probs):
        cumulative += prob
        if r <= cumulative:
            return key
    return keys[-1]

def _random_genome() -> dict[str, dict[str, float]]:
    """Genoma casuale con pesi in [-1, 1]."""
    return {
        phase: {macro: random.uniform(-1.0, 1.0) for macro in macros}
        for phase, macros in SmartPlayer.ALL_PHASES.items()
    }

class RLGA(SmartPlayer):
    """
    Agente Risk con comportamento guidato da un GENOMA evolutivo.

    Il genoma definisce i pesi di preferenza per ogni macro-azione
    in ogni fase. La selezione è softmax (non greedy), quindi rimane
    stocastica ma proporzionale ai valori appresi dall'evoluzione.

    Args:
        color           colore del player
        genome          dizionario pesi (se None → casuale)
        temperature     temperatura softmax (default 1.0)
        troops_to_place truppe iniziali
    """

    def __init__(
        self,
        color: str,
        genome: Optional[dict[str, dict[str, float]]] = None,
        temperature: float = 1.0,
        troops_to_place: int = 0,
    ):
        super().__init__(color, troops_to_place)
        self.genome = genome if genome is not None else _random_genome()
        self.temperature = temperature
        self._cluster: list[Country] | None = None

    #  selezione macro
    def _select_macro(self, phase: str) -> str:
        return _softmax_choice(self.genome[phase], self.temperature)

    #  lifecycle
    def turn_setup(self):
        self._cluster = None

    def action_cleanup(self):
        pass

    #  fasi di gioco
    def place_armies(self) -> Action | None:
        macro = self._select_macro("place")
        return self._execute_place_macro(macro)

    def attack(self) -> Action | None:
        macro = self._select_macro("attack")
        return self._execute_attack_macro(macro)

    def fortify(self) -> Action | None:
        macro = self._select_macro("fortify")
        return self._execute_fortify_macro(macro)

    #  persistenza
    def save(self, filepath: str):
        data = {
            "genome":      self.genome,
            "temperature": self.temperature,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[RLGA {self.color}] salvato in {filepath}")

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.genome = data["genome"]
        self.temperature = data.get("temperature", 1.0)
        print(f"[RLGA {self.color}] caricato da {filepath}")

    #  debug
    def explain(self) -> str:
        lines = [
            f"RLGA [{self.color}] - temperature={self.temperature:.2f}", ""
        ]
        for phase, macros in self.ALL_PHASES.items():
            weights = self.genome[phase]
            best = max(weights, key=weights.__getitem__)
            lines.append(f"  {phase}:")
            for m in macros:
                bar = "█" * max(0, int((weights[m] + 1) * 10))
                mark = " <- BEST" if m == best else ""
                lines.append(f"    {m:<28} {weights[m]:+.3f}  {bar}{mark}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Funzioni genetiche - usate da train_rlga.py
# ---------------------------------------------------------------------------

def crossover(genome_a: dict, genome_b: dict) -> tuple[dict, dict]:
    """
    Crossover uniforme per fase/macro.
    Ogni gene (peso singolo) viene preso a caso da uno dei due genitori.
    Restituisce due figli.
    """
    child1: dict = {}
    child2: dict = {}
    for phase in SmartPlayer.ALL_PHASES:
        child1[phase] = {}
        child2[phase] = {}
        for macro in SmartPlayer.ALL_PHASES[phase]:
            if random.random() < 0.5:
                child1[phase][macro] = genome_a[phase][macro]
                child2[phase][macro] = genome_b[phase][macro]
            else:
                child1[phase][macro] = genome_b[phase][macro]
                child2[phase][macro] = genome_a[phase][macro]
    return child1, child2


def mutate(
    genome: dict,
    mutation_rate: float = 0.15,
    mutation_std:  float = 0.3,
    clip:          float = 3.0,
) -> dict:
    """
    Mutazione gaussiana.
    Ogni gene viene perturbato con probabilità mutation_rate,
    aggiungendo rumore N(0, mutation_std).
    I pesi vengono clampati in [-clip, +clip].
    """
    new_genome: dict = {}
    for phase in SmartPlayer.ALL_PHASES:
        new_genome[phase] = {}
        for macro in SmartPlayer.ALL_PHASES[phase]:
            w = genome[phase][macro]
            if random.random() < mutation_rate:
                w += random.gauss(0.0, mutation_std)
            new_genome[phase][macro] = max(-clip, min(clip, w))
    return new_genome


def tournament_select(
    population: list[dict],
    fitnesses:  list[float],
    k:          int = 3,
) -> dict:
    """
    Selezione torneo: prende k individui a caso e restituisce il migliore.
    """
    indices = random.sample(range(len(population)), k=min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return copy.deepcopy(population[best_idx])
