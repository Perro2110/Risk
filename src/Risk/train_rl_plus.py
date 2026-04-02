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

import glob
import json
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict

from Risk.game import Game
from Risk.map import Map
from Risk.players.stinky import Stinky
from Risk.players.pixie import Pixie
from Risk.players.communist import Communist
from Risk.players.cluster import Cluster
from Risk.players.angry import Angry
from Risk.players.rlph_plus import RLPHPlus

# ── iperparametri ────────────────────────────────────────────────────────────
NUM_EPISODES    = 5000
GAME_LENGTH     = 100
NUM_OPPONENTS   = 2
ROLLING_WINDOW  = 100       # finestra win-rate rolling per il best-checkpoint

BEST_PATH = "rl_best.json"
LAST_PATH = "rl_last.json"
STATS_PATH = "rl_stats.json"
LOG_EVERY  = 10             # stampa a video ogni N episodi
SAVE_EVERY = 100            # flush stats su disco ogni N episodi

# Avversari disponibili
ALL_BOTS: list[type] = [
    Stinky,
    Pixie,
    Communist,
    Cluster,
    Angry,
]


# ── struttura statistiche ─────────────────────────────────────────────────────
@dataclass
class EpisodeStats:
    episode:      int
    won:          bool
    epsilon:      float
    q_states:     int
    # bot-specific: nome bot → vittoria (True/False) per ogni bot in quella partita
    bot_results:  dict[str, bool] = field(default_factory=dict)
    # distribuzione macro per fase: fase → {macro: count}
    macro_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # Q-value medio per macro (sullo stato corrente a fine partita)
    q_snapshot:   dict[str, float] = field(default_factory=dict)


def _snapshot_q(agent: RLPHPlus) -> dict[str, float]:
    """
    Media dei Q-values per ogni macro su tutti gli stati visitati.
    Leggero: O(|Q| * |macros|).
    """
    sums:   dict[str, float] = defaultdict(float)
    counts: dict[str, int]   = defaultdict(int)
    for state_vals in agent.Q.values():
        for macro, val in state_vals.items():
            sums[macro]   += val
            counts[macro] += 1
    return {m: (sums[m] / counts[m] if counts[m] else 0.0) for m in agent.ALL_MACROS}


def _macro_distribution(agent: RLPHPlus) -> dict[str, dict[str, int]]:
    """
    Per ogni fase calcola quante volte ogni macro è la preferita
    (argmax Q) sugli stati visitati di quella fase.
    Restituisce {"place": {macro: n, ...}, "attack": {...}, "fortify": {...}}
    """
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
    # converti defaultdict → dict normale per la serializzazione
    return {ph: dict(v) for ph, v in dist.items()}


# ── ricerca mappa ─────────────────────────────────────────────────────────────
def _find_map_csv() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidates = glob.glob(os.path.join(root, "data", "*.csv"))
    if not candidates:
        raise FileNotFoundError("Nessun CSV trovato in data/.")
    return candidates[0]


# ── training ──────────────────────────────────────────────────────────────────
def run_training():
    map_csv = _find_map_csv()
    print(f"[train] Mappa: {map_csv}")
    print(f"[train] Avversari disponibili: {[b.__name__ for b in ALL_BOTS]}")
    print(f"[train] Episodi: {NUM_EPISODES}  |  GAME_LENGTH: {GAME_LENGTH}\n")

    agent = RLPHPlus("rl")

    # storico episodi e rolling window per best-checkpoint
    history:        list[EpisodeStats] = []
    rolling:        deque[bool]        = deque(maxlen=ROLLING_WINDOW)
    best_win_rate:  float              = -1.0

    # contatori globali per i bot
    bot_wins:   dict[str, int] = defaultdict(int)   # vittorie dei bot
    bot_played: dict[str, int] = defaultdict(int)   # partite in cui erano presenti

    for ep in range(1, NUM_EPISODES + 1):

        # ── scelta avversari ────────────────────────────────────────────────
        bot_classes = random.sample(ALL_BOTS, k=min(NUM_OPPONENTS, len(ALL_BOTS)))
        opponents   = [cls(cls.__name__.lower()) for cls in bot_classes]
        for cls in bot_classes:
            bot_played[cls.__name__] += 1

        # ── costruzione partita ─────────────────────────────────────────────
        game_map = Map.from_csv(map_csv)
        game     = Game(game_map, game_length=GAME_LENGTH, enable_visualizer=False)

        game.add_player(agent)
        for opp in opponents:
            game.add_player(opp)

        game.get_game_state().set_players(game.players)   # type: ignore
        game.play(seed=None)

        # ── analisi risultato ───────────────────────────────────────────────
        alive       = game.get_alive_players()
        agent_won   = any(p is agent for p in alive)
        bot_results: dict[str, bool] = {}

        for opp in opponents:
            opp_alive = any(p is opp for p in alive)
            bot_results[type(opp).__name__] = opp_alive
            if opp_alive:
                bot_wins[type(opp).__name__] += 1

        # ── reward terminale ────────────────────────────────────────────────
        agent.receive_terminal_reward(won=agent_won)

        rolling.append(agent_won)
        rolling_wr = sum(rolling) / len(rolling)

        # ── best checkpoint ─────────────────────────────────────────────────
        if len(rolling) == ROLLING_WINDOW and rolling_wr > best_win_rate:
            best_win_rate = rolling_wr
            agent.save(BEST_PATH)
            print(f"  ★ nuovo best checkpoint  ep={ep}  "
                  f"rolling-wr={best_win_rate*100:.1f}%")

        # ── statistiche episodio ────────────────────────────────────────────
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

        # ── log a video ─────────────────────────────────────────────────────
        if ep % LOG_EVERY == 0:
            total_wins = sum(1 for s in history if s.won)
            # bot-specific win rates (quante partite ha VINTO ogni bot)
            bot_wr_str = "  ".join(
                f"{name}: {bot_wins[name]}/{bot_played[name]}"
                f"({bot_wins[name]/bot_played[name]*100:.0f}%)"
                for name in sorted(bot_played)
            )
            print(
                f"Ep {ep:>5}/{NUM_EPISODES}  "
                f"ε={agent.epsilon:.3f}  "
                f"roll-wr={rolling_wr*100:.1f}%  "
                f"tot-wr={total_wins/ep*100:.1f}%  "
                f"Q-stati={len(agent.Q)}"
            )
            print(f"  bot wins → {bot_wr_str}")

        # ── flush stats su disco ────────────────────────────────────────────
        if ep % SAVE_EVERY == 0:
            _save_stats(history, STATS_PATH)

    # ── fine training ────────────────────────────────────────────────────────
    agent.save(LAST_PATH)
    _save_stats(history, STATS_PATH)

    total_wins = sum(1 for s in history if s.won)
    print(f"\n{'='*60}")
    print(f"Training completo.")
    print(f"  Win-rate totale : {total_wins/NUM_EPISODES*100:.1f}%")
    print(f"  Best rolling-wr : {best_win_rate*100:.1f}%")
    print(f"  Q-stati finali  : {len(agent.Q)}")
    print(f"  Best checkpoint : {BEST_PATH}")
    print(f"  Last checkpoint : {LAST_PATH}")
    print(f"  Stats           : {STATS_PATH}")
    print(f"{'='*60}\n")

    # ── riepilogo bot ────────────────────────────────────────────────────────
    print("Win-rate per bot avversario:")
    for name in sorted(bot_played):
        n = bot_played[name]
        w = bot_wins[name]
        print(f"  {name:<20} {w:>4}/{n:<4}  ({w/n*100:5.1f}% vittorie bot)")

    # ── riepilogo macro preferite ────────────────────────────────────────────
    if history:
        last_dist = history[-1].macro_counts
        print("\nMacro preferite (argmax Q) a fine training:")
        for phase, counts in last_dist.items():
            if counts:
                best = max(counts, key=counts.__getitem__)
                print(f"  {phase:<8}: {best} ({counts[best]} stati)")

    print()
    _print_q_summary(history)


# ── utilità ───────────────────────────────────────────────────────────────────
def _save_stats(history: list[EpisodeStats], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in history], f, indent=2)


def _print_q_summary(history: list[EpisodeStats]):
    """Stampa i Q-value medi finali ordinati per macro."""
    if not history:
        return
    snap = history[-1].q_snapshot
    if not snap:
        return
    print("Q-value medi a fine training (tutte le macro, tutti gli stati):")
    for macro, val in sorted(snap.items(), key=lambda kv: -kv[1]):
        bar = "█" * max(0, int((val + 5) * 2))   # scala approssimativa
        print(f"  {macro:<25} {val:+7.3f}  {bar}")


if __name__ == "__main__":
    run_training()