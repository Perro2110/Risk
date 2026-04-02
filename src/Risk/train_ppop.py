"""
train_menny.py - Training loop per PPOPlayer.

Lancialo dalla ROOT del progetto con:

    python -m Risk.train_menny

Al termine salva l'ensemble in menny_ensemble.json nella root.
"""

import contextlib
import glob
import io
import os
import random
from collections import deque

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
from Risk.players.angry import Angry
from Risk.players.cluster import Cluster
from Risk.players.communist import Communist
from Risk.players.pixie import Pixie
from Risk.players.stinky import Stinky
from Risk.players.manny2 import PPOPlayer


# ── config ──────────────────────────────────────────────────────────────────
NUM_EPISODES   = 5000
GAME_LENGTH    = 150
NUM_OPPONENTS  = 2
SAVE_PATH      = "menny_ensemble.json"
SAVE_EVERY     = 50
LOG_LINES      = 16

ALL_BOTS = [
    Pixie,
    Cluster,
    # Angry,
    Communist,
    # Stinky
]


# ── helpers ─────────────────────────────────────────────────────────────────
def _find_map_csv() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidates = glob.glob(os.path.join(root, "data", "*.csv"))
    if not candidates:
        raise FileNotFoundError("Nessun CSV trovato in data/.")
    return candidates[0]


def _win_rate(wins: int, ep: int) -> float:
    return wins / ep * 100 if ep else 0.0


def _wr_color(wr: float) -> str:
    if wr >= 30:
        return "green"
    if wr >= 10:
        return "yellow"
    return "red"


# ── UI ───────────────────────────────────────────────────────────────────────
def _stats_table(
    ep: int,
    wins: int,
    agent: PPOPlayer,
    bot_wins: dict[str, int],
) -> Table:
    t = Table.grid(padding=(0, 3))
    t.add_column(style="bold", min_width=18)
    t.add_column(min_width=14)
    t.add_column(style="bold", min_width=18)
    t.add_column(min_width=14)

    wr = _win_rate(wins, ep)
    wr_color = _wr_color(wr)

    t.add_row(
        "Episodio",    f"{ep} / {NUM_EPISODES}",
        "Win-rate",    Text(f"{wr:.1f}%", style=wr_color),
    )

    t.add_row("", "", "", "")
    t.add_row("[dim]avversario[/dim]", "[dim]vittorie[/dim]",
              "[dim]win%[/dim]",       "")

    for name, bw in sorted(bot_wins.items(), key=lambda x: -x[1]):
        bwr = _win_rate(bw, ep)
        t.add_row(
            f"  {name}", str(bw),
            Text(f"{bwr:.1f}%", style=_wr_color(bwr)), "",
        )

    return t


def _render(
    ep: int,
    wins: int,
    agent: PPOPlayer,
    bot_wins: dict[str, int],
    log: deque,
    progress: Progress,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top",      size=7 + len(bot_wins)),
        Layout(name="bottom",   size=LOG_LINES + 2),
        Layout(name="progress", size=3),
    )

    layout["top"].split_row(
        Layout(name="stats",  ratio=2),
        Layout(name="macros", ratio=1),
    )

    layout["stats"].update(Panel(
        _stats_table(max(ep, 1), wins, agent, bot_wins),
        title="[bold]PPOPlayer Training[/bold]",
        border_style="blue",
    ))
    layout["bottom"].update(Panel(
        Text.from_markup("\n".join(log)),
        title="[dim]log[/dim]",
        border_style="dim",
    ))
    layout["progress"].update(progress)
    return layout


# ── training loop ────────────────────────────────────────────────────────────
def run_training():
    map_csv = _find_map_csv()
    console = Console()

    agent = PPOPlayer("ppop")

    wins = 0
    bot_wins: dict[str, int] = {b.__name__: 0 for b in ALL_BOTS}

    log: deque[str] = deque(maxlen=LOG_LINES)
    log.append(f"[dim]mappa: {os.path.basename(map_csv)}[/dim]")
    log.append(f"[dim]avversari disponibili: {[b.__name__ for b in ALL_BOTS]}[/dim]")
    log.append(f"[dim]{NUM_EPISODES} episodi · game_length={GAME_LENGTH} · {NUM_OPPONENTS} opponent/partita[/dim]")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )
    task = progress.add_task("Training PPOPlayer", total=NUM_EPISODES)

    with Live(
        _render(0, 0, agent, bot_wins, log, progress),
        console=console,
        refresh_per_second=4,
        vertical_overflow="visible",
    ) as live:

        for ep in range(1, NUM_EPISODES + 1):

            # costruisci la partita
            game_map = Map.from_csv(map_csv)
            game = Game(game_map, game_length=GAME_LENGTH, enable_visualizer=False)

            k = min(NUM_OPPONENTS, len(ALL_BOTS))
            bot_classes = random.sample(ALL_BOTS, k=k)
            opponents = [cls(cls.__name__.lower()) for cls in bot_classes]

            game.add_player(agent)
            for opp in opponents:
                game.add_player(opp)

            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                game.play(seed=None)

            # determina il vincitore
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

            # reward terminale a PPOPlayer
            agent.receive_terminal_reward(agent_won)

            if agent_won:
                wins += 1
            else:
                for bc in bot_classes:
                    if any(isinstance(p, bc) for p in alive if p is not agent):
                        bot_wins[bc.__name__] += 1

            # log ogni 10 episodi
            if ep % 10 == 0:
                opp_str = ", ".join(type(o).__name__ for o in opponents)
                wr = _win_rate(wins, ep)
                result = "[green]WIN[/green]" if agent_won else "[red]loss[/red]"
                log.append(
                    f"ep {ep:>5}  wr={wr:.1f}%  "
                    f"{result}  vs [{opp_str}]"
                )

            progress.advance(task)
            live.update(_render(ep, wins, agent, bot_wins, log, progress))

    # ── riepilogo finale ─────────────────────────────────────────────────────
    console.print()
    console.print("[bold]══ RISULTATI FINALI ══[/bold]")
    console.print(
        f"  PPOPlayer win-rate : [bold green]{_win_rate(wins, NUM_EPISODES):.1f}%[/bold green]"
        f"  ({wins}/{NUM_EPISODES})"
    )
    console.print()
    console.print("  Win-rate avversari (quando vincono):")
    for name, bw in sorted(bot_wins.items(), key=lambda x: -x[1]):
        bwr = _win_rate(bw, NUM_EPISODES)
        color = _wr_color(bwr)
        console.print(f"    {name:<14} {Text(f'{bwr:.1f}%', style=color)}  ({bw}/{NUM_EPISODES})")
    console.print()
    console.print(f"  Ensemble salvato in [bold]{SAVE_PATH}[/bold]")


if __name__ == "__main__":
    run_training()
