"""
train_rl.py - Training loop per RLPH.

Lancialo dalla ROOT del progetto con:

    python -m Risk.train_rl

Al termine salva la Q-table in rl_qtable.json nella root.
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
from Risk.players.manny import MENNY    
from Risk.players.rlph import RLPH
from Risk.players.stinky import Stinky


NUM_EPISODES = 5000             # partite di training
GAME_LENGTH = 100               # turni massimi per partita
NUM_OPPONENTS = 2               # numero di opponents per ogni partita
EPSILON_START = 0.8             # esplorazione iniziale
EPSILON_END = 0.8               # esplorazione finale
QTABLE_PATH = "rl_qtable.json"  # salvato nella root del progetto
SAVE_EVERY = 50                 # salva ogni N episodi
LOG_LINES = 14

EPSILON_DECAY = (EPSILON_START - EPSILON_END) / NUM_EPISODES

ALL_BOTS = [Pixie] # , Cluster]
BOT_WINS = [0] * len(ALL_BOTS)


def _find_map_csv() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidates = glob.glob(os.path.join(root, "data", "*.csv"))
    if not candidates:
        raise FileNotFoundError("No CSV found in data/.")
    return candidates[0]


def _stats_table(ep: int, wins: int, q_states: int, epsilon: float) -> Table:
    t = Table.grid(padding=(0, 3))
    t.add_column(style="bold", min_width=16)
    t.add_column(min_width=12)
    t.add_column(style="bold", min_width=16)
    t.add_column(min_width=12)

    win_rate = wins / ep * 100 if ep else 0.0
    wr_color = "green" if win_rate >= 10 \
        else "yellow" if win_rate > 0 else "red"

    t.add_row("Episode",     f"{ep} / {NUM_EPISODES}",
              "Win-rate",    Text(f"{win_rate:.1f}%", style=wr_color))
    t.add_row("epsilon",     f"{epsilon:.3f}",
              "Q-states",    str(q_states))
    t.add_row("RLPH wins",   str(wins), "", "")
    for i, cls in enumerate(ALL_BOTS):
        t.add_row(f"  {cls.__name__}", str(BOT_WINS[i]), "", "")
    return t


def _render(ep: int, wins: int, agent: RLPH,
            log: deque, progress: Progress) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="stats",    size=5 + len(ALL_BOTS)),
        Layout(name="log",      size=LOG_LINES + 2),
        Layout(name="progress", size=3),
    )
    layout["stats"].update(Panel(
        _stats_table(max(ep, 1), wins, len(agent.Q), agent.epsilon),
        title="[bold]RLPH Training[/bold]", border_style="blue",
    ))
    layout["log"].update(Panel(
        Text.from_markup("\n".join(log)),
        title="[dim]log[/dim]", border_style="dim",
    ))
    layout["progress"].update(progress)
    return layout


def run_training():
    map_csv = _find_map_csv()
    console = Console()

    agent = RLPH("rl", epsilon=EPSILON_START)
    wins = 0

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
    task = progress.add_task("Training RLPH", total=NUM_EPISODES)

    with Live(_render(0, 0, agent, log, progress),
              console=console, refresh_per_second=4,
              vertical_overflow="visible") as live:

        for ep in range(1, NUM_EPISODES + 1):
            agent.epsilon = max(EPSILON_END,
                                EPSILON_START - EPSILON_DECAY * ep)

            game_map = Map.from_csv(map_csv)
            game = Game(game_map, game_length=GAME_LENGTH,
                        enable_visualizer=False)

            k = min(NUM_OPPONENTS, len(ALL_BOTS))
            bot_classes = random.sample(ALL_BOTS, k=k)
            opponents = [cls(cls.__name__.lower()) for cls in bot_classes]

            game.add_player(agent)
            for opp in opponents:
                game.add_player(opp)

            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                game.play(seed=None)

            # Determine winner (game.py now calls receive_terminal_reward too)
            alive = game.get_alive_players()
            if len(alive) == 1:
                agent_won = alive[0] is agent
            else:
                best = max(
                    alive,
                    key=lambda p: len(
                        game.get_game_state()
                            .get_game_map().get_owned_countries(p)
                    )
                )
                agent_won = best is agent

            if agent_won:
                wins += 1
            else:
                for i, b in enumerate(ALL_BOTS):
                    if any(isinstance(p, b) for p in alive if p is not agent):
                        BOT_WINS[i] += 1

            if ep % 10 == 0:
                opp_str = ", ".join(type(o).__name__ for o in opponents)
                result = "[green]WIN[/green]" if agent_won else "[red]loss[/red]"
                log.append(
                    f"ep {ep:>5}  e={agent.epsilon:.3f}  "
                    f"{result}  vs [{opp_str}]  Q={len(agent.Q)}"
                )

            if ep % SAVE_EVERY == 0:
                with contextlib.redirect_stdout(io.StringIO()):
                    agent.save(QTABLE_PATH)
                log.append(
                    f"[dim]  -> saved Q-table ({len(agent.Q)} states)[/dim]"
                )

            progress.advance(task)
            live.update(_render(ep, wins, agent, log, progress))

    agent.save(QTABLE_PATH)
    console.print(
        f"\n[bold green]Training done.[/bold green] "
        f"Win-rate: {wins / NUM_EPISODES * 100:.1f}%  "
        f"Final Q-states: {len(agent.Q)}"
    )


if __name__ == "__main__":
    run_training()
