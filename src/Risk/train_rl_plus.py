"""
train_rlph_plus.py — Training loop per RLPHPlus.

Lancialo dalla ROOT del progetto con:
    python -m Risk.train_rlph_plus

Output:
    rl_best.json      — checkpoint con il miglior win-rate rolling (ultimi 100 ep)
    rl_last.json      — stato finale a fine training
    rl_stats.json     — storico completo per analisi/plot
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import sys
import os
import random
from collections import defaultdict, deque
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
from Risk.players.deepraph import DeepRLPH
from Risk.players.rlph_plus import RLPHPlus

# ── iperparametri ────────────────────────────────────────────────────────────
NUM_EPISODES   = 5000
GAME_LENGTH    = 300
NUM_OPPONENTS  = 2          # un solo avversario per partita (come train_rl.py)
ROLLING_WINDOW = 100        # finestra win-rate rolling per il best-checkpoint

BEST_PATH  = "rl_best.json"
LAST_PATH  = "rl_last.json"
STATS_PATH = "rl_stats.json"
SAVE_EVERY = 50             # flush stats + save ogni N episodi
LOG_LINES  = 14

# Avversari disponibili
ALL_BOTS: list[type] = [
    # RLPH,
    Pixie,
    Communist,
    Cluster,
    # Angry,
]

# contatori globali bot (paralleli ad ALL_BOTS)
BOT_WINS: list[int] = [0] * len(ALL_BOTS)


# ── struttura statistiche ─────────────────────────────────────────────────────
@dataclass
class EpisodeStats:
    episode:      int
    won:          bool
    epsilon:      float
    q_states:     int
    bot_results:  dict[str, bool]         = field(default_factory=dict)
    macro_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    q_snapshot:   dict[str, float]        = field(default_factory=dict)


def _snapshot_q(agent: RLPHPlus) -> dict[str, float]:
    sums:   dict[str, float] = defaultdict(float)
    counts: dict[str, int]   = defaultdict(int)
    for state_vals in agent.Q.values():
        for macro, val in state_vals.items():
            sums[macro]   += val
            counts[macro] += 1
    return {m: (sums[m] / counts[m] if counts[m] else 0.0) for m in agent.ALL_MACROS}


def _macro_distribution(agent: RLPHPlus) -> dict[str, dict[str, int]]:
    PHASE_NAMES = {0: "place", 1: "attack", 2: "fortify"}
    dist: dict[str, dict[str, int]] = {
        "place":   defaultdict(int),
        "attack":  defaultdict(int),
        "fortify": defaultdict(int),
    }
    for state, vals in agent.Q.items():
        if len(state) != 5:
            continue
        phase_name = PHASE_NAMES.get(state[4])
        if phase_name is None:
            continue
        best = max(vals, key=vals.__getitem__)
        dist[phase_name][best] += 1
    return {ph: dict(v) for ph, v in dist.items()}


# ── ricerca mappa ─────────────────────────────────────────────────────────────
def _find_map_csv() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidates = glob.glob(os.path.join(root, "data", "*.csv"))
    if not candidates:
        raise FileNotFoundError("Nessun CSV trovato in data/.")
    return candidates[0]


# ── UI helpers ────────────────────────────────────────────────────────────────
def _stats_table(ep: int, wins: int, agent: RLPHPlus,
                 rolling_wr: float, best_wr: float) -> Table:
    t = Table.grid(padding=(0, 3))
    t.add_column(style="bold", min_width=16)
    t.add_column(min_width=12)
    t.add_column(style="bold", min_width=16)
    t.add_column(min_width=12)

    win_rate = wins / ep * 100 if ep else 0.0
    wr_color = "green" if win_rate >= 10 else "yellow" if win_rate > 0 else "red"
    rw_color = "green" if rolling_wr >= 0.1 else "yellow" if rolling_wr > 0 else "red"

    t.add_row("Episode",    f"{ep} / {NUM_EPISODES}",
              "Win-rate",   Text(f"{win_rate:.1f}%", style=wr_color))
    t.add_row("epsilon",    f"{agent.epsilon:.3f}",
              "Roll-WR",    Text(f"{rolling_wr*100:.1f}%", style=rw_color))
    t.add_row("Best-WR",    f"{best_wr*100:.1f}%",
              "Q-states",   str(len(agent.Q)))
    for i, cls in enumerate(ALL_BOTS):
        t.add_row(f"  {cls.__name__}", str(BOT_WINS[i]), "", "")
    return t


def _render(ep: int, wins: int, agent: RLPHPlus,
            rolling_wr: float, best_wr: float,
            log: deque, progress: Progress) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="stats",    size=6 + len(ALL_BOTS)),
        Layout(name="log",      size=LOG_LINES + 2),
        Layout(name="progress", size=3),
    )
    layout["stats"].update(Panel(
        _stats_table(max(ep, 1), wins, agent, rolling_wr, best_wr),
        title="[bold]RLPHPlus Training[/bold]", border_style="blue",
    ))
    layout["log"].update(Panel(
        Text.from_markup("\n".join(log)),
        title="[dim]log[/dim]", border_style="dim",
    ))
    layout["progress"].update(progress)
    return layout


# ── serializzazione stats ─────────────────────────────────────────────────────
def _save_stats(history: list[EpisodeStats], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in history], f, indent=2)


# ── training ──────────────────────────────────────────────────────────────────
def run_training():
    map_csv = _find_map_csv()
    console = Console()

    agent = DeepRLPH("rl")

    wins:          int              = 0
    history:       list[EpisodeStats] = []
    rolling:       deque[bool]      = deque(maxlen=ROLLING_WINDOW)
    best_win_rate: float            = -1.0
    rolling_wr:    float            = 0.0

    log: deque[str] = deque(maxlen=LOG_LINES)
    log.append(f"[dim]map: {os.path.basename(map_csv)}[/dim]")
    log.append(f"[dim]opponents: {[b.__name__ for b in ALL_BOTS]}[/dim]")
    log.append(f"[dim]{NUM_EPISODES} episodes  game_length={GAME_LENGTH}[/dim]")

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
    task = progress.add_task("Training RLPHPlus", total=NUM_EPISODES)

    with Live(_render(0, 0, agent, 0.0, 0.0, log, progress),
              console=console, refresh_per_second=4,
              vertical_overflow="visible") as live:

        for ep in range(1, NUM_EPISODES + 1):

            # ── scelta avversari ────────────────────────────────────────────
            bot_classes = random.sample(ALL_BOTS, k=min(NUM_OPPONENTS, len(ALL_BOTS)))
            opponents = [cls(cls.__name__.lower()) for cls in bot_classes]

            # ── costruzione partita ─────────────────────────────────────────
            game_map = Map.from_csv(map_csv)
            game     = Game(game_map, game_length=GAME_LENGTH, enable_visualizer=False)

            game.add_player(agent)
            for opp in opponents:
                game.add_player(opp)

            seed = random.randrange(sys.maxsize)
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                game.play(seed)

            # ── analisi risultato ───────────────────────────────────────────
            alive = game.get_alive_players()
            if len(alive) == 1:
                agent_won = alive[0] is agent
            else:
                best = max(
                    alive,
                    key=lambda p: len(
                        game.get_game_state()
                            .get_game_map().get_owned_countries(p)
                    ),
                )
                agent_won = best is agent

            if agent_won:
                wins += 1
            else:
                for i, b in enumerate(ALL_BOTS):
                    if any(isinstance(p, b) for p in alive if p is not agent):
                        BOT_WINS[i] += 1

            bot_results: dict[str, bool] = {
                type(opp).__name__: any(p is opp for p in alive)
                for opp in opponents
            }

            # ── reward terminale ────────────────────────────────────────────
            agent.receive_terminal_reward(won=agent_won)

            rolling.append(agent_won)
            rolling_wr = sum(rolling) / len(rolling)

            # ── best checkpoint ─────────────────────────────────────────────
            if len(rolling) == ROLLING_WINDOW and rolling_wr > best_win_rate:
                best_win_rate = rolling_wr
                with contextlib.redirect_stdout(io.StringIO()):
                    agent.save(BEST_PATH)
                log.append(
                    f"[bold yellow]★ best checkpoint[/bold yellow]  "
                    f"ep={ep}  roll-wr={best_win_rate*100:.1f}%"
                )

            # ── statistiche episodio ────────────────────────────────────────
            stats = EpisodeStats(
                episode      = ep,
                won          = agent_won,
                epsilon      = agent.epsilon,
                q_states     = len(agent.Q),
                bot_results  = bot_results,
                macro_counts = _macro_distribution(agent),
                q_snapshot   = _snapshot_q(agent),
            )
            history.append(stats)

            # ── log a video ogni 10 episodi ─────────────────────────────────
            if ep % 10 == 0:
                opp_str = ", ".join(type(o).__name__ for o in opponents)
                result  = "[green]WIN[/green]" if agent_won else "[red]loss[/red]"
                log.append(
                    f"ep {ep:>5}  e={agent.epsilon:.3f}  "
                    f"{result}  vs [{opp_str}]  Q={len(agent.Q)}"
                )

            # ── flush su disco ──────────────────────────────────────────────
            if ep % SAVE_EVERY == 0:
                with contextlib.redirect_stdout(io.StringIO()):
                    agent.save(LAST_PATH)
                _save_stats(history, STATS_PATH)
                log.append(
                    f"[dim]  -> saved ({len(agent.Q)} Q-states)[/dim]"
                )

            progress.advance(task)
            live.update(_render(ep, wins, agent, rolling_wr, best_win_rate, log, progress))

    # ── fine training ────────────────────────────────────────────────────────
    with contextlib.redirect_stdout(io.StringIO()):
        agent.save(LAST_PATH)
    _save_stats(history, STATS_PATH)

    total_wins = sum(1 for s in history if s.won)
    console.print(
        f"\n[bold green]Training completo.[/bold green]  "
        f"Win-rate: {total_wins/NUM_EPISODES*100:.1f}%  "
        f"Best rolling-WR: {best_win_rate*100:.1f}%  "
        f"Q-stati: {len(agent.Q)}"
    )

    console.print("\nWin-rate per bot avversario:")
    for i, cls in enumerate(ALL_BOTS):
        n = sum(1 for s in history if cls.__name__ in s.bot_results)
        w = BOT_WINS[i]
        if n:
            console.print(f"  {cls.__name__:<20} {w:>4}/{n:<4}  ({w/n*100:5.1f}% vittorie bot)")

    if history:
        last_dist = history[-1].macro_counts
        console.print("\nMacro preferite (argmax Q) a fine training:")
        for phase, counts in last_dist.items():
            if counts:
                best_macro = max(counts, key=counts.__getitem__)
                console.print(f"  {phase:<8}: {best_macro} ({counts[best_macro]} stati)")

    snap = history[-1].q_snapshot if history else {}
    if snap:
        console.print("\nQ-value medi a fine training:")
        for macro, val in sorted(snap.items(), key=lambda kv: -kv[1]):
            bar = "█" * max(0, int((val + 5) * 2))
            console.print(f"  {macro:<25} {val:+7.3f}  {bar}")


if __name__ == "__main__":
    run_training()