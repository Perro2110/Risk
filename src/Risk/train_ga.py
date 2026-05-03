"""
train_rlga.py - Training loop con Algoritmo Genetico per RLGA.

Lancialo dalla ROOT del progetto con:
    python -m Risk.train_rlga

Output:
    ga_best.json      - genoma del miglior individuo trovato
    ga_last.json      - genoma dell'ultimo best (fine training)
    ga_stats.json     - storico generazioni per analisi/plot

Algoritmo:
    1. Inizializza popolazione casuale
    2. Per ogni generazione:
       a. Valuta fitness di ogni individuo (win-rate su EVAL_GAMES partite)
       b. Élitismo: copia il best direttamente nella nuova generazione
       c. Riempi il resto con figli (torneo → crossover → mutazione)
    3. Salva best e stats ogni SAVE_EVERY generazioni
"""

from __future__ import annotations

import contextlib
import copy
import glob
import io
import json
import os
import random
from collections import deque
from dataclasses import dataclass, field, asdict

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from Risk.game import Game
from Risk.map import Map
from Risk.players.stinky import Stinky
from Risk.players.pixie import Pixie
from Risk.players.communist import Communist
from Risk.players.cluster import Cluster
from Risk.players.angry import Angry
from Risk.players.rlga import RLGA, _random_genome, crossover, mutate, tournament_select

# ── iperparametri GA ─────────────────────────────────────────────────────────
POP_SIZE       = 30       # individui nella popolazione
NUM_GENERATIONS = 20      # generazioni di evoluzione
EVAL_GAMES     = 20       # partite per valutare la fitness di un individuo
GAME_LENGTH    = 100      # lunghezza massima partita
NUM_OPPONENTS  = 3        # avversari per partita

ELITE_N        = 2        # quanti individui passano direttamente (élitismo)
TOURNAMENT_K   = 3        # dimensione torneo
MUTATION_RATE  = 0.15     # probabilità mutazione per gene
MUTATION_STD   = 0.30     # deviazione standard rumore gaussiano
TEMPERATURE    = 1.0      # temperatura softmax per la selezione macro

BEST_PATH  = "ga_best.json"
LAST_PATH  = "ga_last.json"
STATS_PATH = "ga_stats.json"
SAVE_EVERY = 10           # salva ogni N generazioni
LOG_LINES  = 16

ALL_BOTS: list[type] = [
    Stinky,
    Pixie,
    Communist,
    Cluster,
    Angry,
]


# ── struttura statistiche ─────────────────────────────────────────────────────
@dataclass
class GenerationStats:
    generation:   int
    best_fitness: float
    avg_fitness:  float
    worst_fitness: float
    best_genome:  dict = field(default_factory=dict)


def _save_stats(history: list[GenerationStats], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in history], f, indent=2)


# ── ricerca mappa ─────────────────────────────────────────────────────────────
def _find_map_csv() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidates = glob.glob(os.path.join(root, "data", "*.csv"))
    if not candidates:
        raise FileNotFoundError("Nessun CSV trovato in data/.")
    return candidates[0]


# ── valutazione fitness ───────────────────────────────────────────────────────
def evaluate_fitness(genome: dict, map_csv: str, n_games: int = EVAL_GAMES) -> float:
    """
    Gioca n_games partite con questo genoma contro avversari casuali.
    Fitness = win-rate (0.0 – 1.0).
    """
    wins = 0
    for _ in range(n_games):
        agent    = RLGA("ga_agent", genome=copy.deepcopy(genome), temperature=TEMPERATURE)
        bot_cls  = random.choice(ALL_BOTS)
        opponent = bot_cls(bot_cls.__name__.lower())

        game_map = Map.from_csv(map_csv)
        game     = Game(game_map, game_length=GAME_LENGTH, enable_visualizer=False)
        game.add_player(agent)
        game.add_player(opponent)

        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            game.play(seed=None)

        alive = game.get_alive_players()
        if len(alive) == 1:
            won = alive[0] is agent
        else:
            gm  = game.get_game_state().get_game_map()
            best_p = max(alive, key=lambda p: len(gm.get_owned_countries(p)))
            won = best_p is agent

        if won:
            wins += 1

    return wins / n_games


# ── UI ────────────────────────────────────────────────────────────────────────
def _stats_table(gen: int, best_f: float, avg_f: float,
                 overall_best: float, pop_size: int) -> Table:
    t = Table.grid(padding=(0, 3))
    t.add_column(style="bold", min_width=18)
    t.add_column(min_width=12)
    t.add_column(style="bold", min_width=18)
    t.add_column(min_width=12)

    bf_color = "green" if best_f >= 0.3 else "yellow" if best_f >= 0.1 else "red"
    ob_color = "green" if overall_best >= 0.3 else "yellow" if overall_best >= 0.1 else "red"

    t.add_row("Generation",    f"{gen} / {NUM_GENERATIONS}",
              "Best fitness",  Text(f"{best_f*100:.1f}%",     style=bf_color))
    t.add_row("Avg fitness",   f"{avg_f*100:.1f}%",
              "Overall best",  Text(f"{overall_best*100:.1f}%", style=ob_color))
    t.add_row("Pop size",      str(pop_size),
              "Eval games/ind",str(EVAL_GAMES))
    return t


def _render(gen: int, best_f: float, avg_f: float, overall_best: float,
            log: deque, progress: Progress) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="stats",    size=7),
        Layout(name="log",      size=LOG_LINES + 2),
        Layout(name="progress", size=3),
    )
    layout["stats"].update(Panel(
        _stats_table(max(gen, 1), best_f, avg_f, overall_best, POP_SIZE),
        title="[bold]RLGA - Algoritmo Genetico[/bold]", border_style="green",
    ))
    layout["log"].update(Panel(
        Text.from_markup("\n".join(log)),
        title="[dim]log[/dim]", border_style="dim",
    ))
    layout["progress"].update(progress)
    return layout


def _genome_summary(genome: dict) -> str:
    """Una riga riassuntiva del genoma best."""
    parts = []
    for phase, weights in genome.items():
        best_macro = max(weights, key=weights.__getitem__)
        parts.append(f"{phase[0]}:{best_macro.split('_', 1)[1][:8]}")
    return "  ".join(parts)


# ── training GA ───────────────────────────────────────────────────────────────
def run_training():
    map_csv = _find_map_csv()
    console = Console()

    # ── inizializza popolazione ──────────────────────────────────────────────
    population: list[dict]  = [_random_genome() for _ in range(POP_SIZE)]
    fitnesses:  list[float] = [0.0] * POP_SIZE

    history:      list[GenerationStats] = []
    overall_best_fitness: float         = -1.0
    overall_best_genome:  dict | None   = None

    log: deque[str] = deque(maxlen=LOG_LINES)
    log.append(f"[dim]map: {os.path.basename(map_csv)}[/dim]")
    log.append(f"[dim]pop={POP_SIZE}  gen={NUM_GENERATIONS}  eval={EVAL_GAMES}  opp={[b.__name__ for b in ALL_BOTS]}[/dim]")
    log.append(f"[dim]mutation_rate={MUTATION_RATE}  std={MUTATION_STD}  elites={ELITE_N}[/dim]")

    total_evals = NUM_GENERATIONS * POP_SIZE

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console, expand=True,
    )
    task = progress.add_task("Evoluzione GA", total=total_evals)

    with Live(_render(0, 0.0, 0.0, 0.0, log, progress),
              console=console, refresh_per_second=2,
              vertical_overflow="visible") as live:

        for gen in range(1, NUM_GENERATIONS + 1):

            # ── valutazione fitness ──────────────────────────────────────────
            for idx in range(POP_SIZE):
                fitnesses[idx] = evaluate_fitness(population[idx], map_csv)
                progress.advance(task)
                live.update(_render(
                    gen,
                    max(fitnesses),
                    sum(fitnesses) / len(fitnesses),
                    overall_best_fitness,
                    log, progress,
                ))

            best_idx    = max(range(POP_SIZE), key=lambda i: fitnesses[i])
            best_f      = fitnesses[best_idx]
            avg_f       = sum(fitnesses) / POP_SIZE
            worst_f     = min(fitnesses)

            # ── aggiorna best globale ────────────────────────────────────────
            if best_f > overall_best_fitness:
                overall_best_fitness = best_f
                overall_best_genome  = copy.deepcopy(population[best_idx])
                summary = _genome_summary(overall_best_genome)
                log.append(
                    f"[bold yellow]★ nuovo best[/bold yellow]  "
                    f"gen={gen}  fitness={best_f*100:.1f}%  [{summary}]"
                )
                # salva subito il best
                with open(BEST_PATH, "w", encoding="utf-8") as f:
                    json.dump({"genome": overall_best_genome,
                               "fitness": overall_best_fitness,
                               "generation": gen,
                               "temperature": TEMPERATURE}, f, indent=2)

            # ── stats generazione ────────────────────────────────────────────
            history.append(GenerationStats(
                generation    = gen,
                best_fitness  = best_f,
                avg_fitness   = avg_f,
                worst_fitness = worst_f,
                best_genome   = copy.deepcopy(population[best_idx]),
            ))

            # ── log ogni generazione ─────────────────────────────────────────
            log.append(
                f"gen {gen:>4}  best={best_f*100:5.1f}%  "
                f"avg={avg_f*100:5.1f}%  worst={worst_f*100:5.1f}%"
            )

            # ── nuova generazione ────────────────────────────────────────────
            # ordina per fitness decrescente
            sorted_pairs = sorted(
                zip(fitnesses, population),
                key=lambda x: x[0],
                reverse=True,
            )
            sorted_pop = [g for _, g in sorted_pairs]
            sorted_fit = [f for f, _ in sorted_pairs]

            new_population: list[dict] = []

            # élitismo
            for i in range(min(ELITE_N, POP_SIZE)):
                new_population.append(copy.deepcopy(sorted_pop[i]))

            # figli tramite selezione torneo + crossover + mutazione
            while len(new_population) < POP_SIZE:
                parent_a = tournament_select(sorted_pop, sorted_fit, k=TOURNAMENT_K)
                parent_b = tournament_select(sorted_pop, sorted_fit, k=TOURNAMENT_K)
                child_a, child_b = crossover(parent_a, parent_b)
                child_a = mutate(child_a, MUTATION_RATE, MUTATION_STD)
                child_b = mutate(child_b, MUTATION_RATE, MUTATION_STD)
                new_population.append(child_a)
                if len(new_population) < POP_SIZE:
                    new_population.append(child_b)

            population = new_population
            fitnesses  = [0.0] * POP_SIZE

            # ── salvataggio periodico ────────────────────────────────────────
            if gen % SAVE_EVERY == 0:
                _save_stats(history, STATS_PATH)
                if overall_best_genome:
                    with open(LAST_PATH, "w", encoding="utf-8") as f:
                        json.dump({"genome": overall_best_genome,
                                   "fitness": overall_best_fitness,
                                   "generation": gen,
                                   "temperature": TEMPERATURE}, f, indent=2)
                log.append(f"[dim]  -> stats salvate (gen {gen})[/dim]")

            live.update(_render(
                gen, best_f, avg_f, overall_best_fitness, log, progress
            ))

    # ── fine training ─────────────────────────────────────────────────────────
    _save_stats(history, STATS_PATH)
    if overall_best_genome:
        with open(LAST_PATH, "w", encoding="utf-8") as f:
            json.dump({"genome": overall_best_genome,
                       "fitness": overall_best_fitness,
                       "temperature": TEMPERATURE}, f, indent=2)

    console.print(
        f"\n[bold green]Training GA completo.[/bold green]  "
        f"Miglior fitness: {overall_best_fitness*100:.1f}%"
    )

    if overall_best_genome:
        console.print("\nGenoma migliore trovato:")
        for phase, weights in overall_best_genome.items():
            best_macro = max(weights, key=weights.__getitem__)
            console.print(f"\n  [bold]{phase}[/bold]:")
            for macro in RLGA.ALL_PHASES[phase]:
                bar  = "█" * max(0, int((weights[macro] + 3) * 5))
                mark = " ◄" if macro == best_macro else ""
                console.print(f"    {macro:<28} {weights[macro]:+.3f}  {bar}{mark}")

    console.print("\nCurva fitness per generazione:")
    for s in history[::max(1, len(history)//20)]:
        bar = "█" * int(s.best_fitness * 40)
        console.print(
            f"  gen {s.generation:>4}  "
            f"best={s.best_fitness*100:5.1f}%  avg={s.avg_fitness*100:5.1f}%  {bar}"
        )


if __name__ == "__main__":
    run_training()