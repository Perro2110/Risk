"""
train_rl.py - Training loop per RLPlayer.

Metti questo file in src/Risk/ insieme agli altri.
Lancialo dalla ROOT del progetto con:

    python -m Risk.train_rl

Al termine salva la Q-table in rl_qtable.json nella root.
Per usarla in main.py:

    from Risk.rl_player import RLPlayer
    agent = RLPlayer("green", epsilon=0.0)
    agent.load("rl_qtable.json")
"""

import glob
import os
import random

from Risk.game import Game
from Risk.map import Map
from Risk.players.stinky import Stinky
from Risk.players.pixie import Pixie
from Risk.players.communist import Communist
from Risk.players.cluster import Cluster
from Risk.players.angry import Angry
from Risk.players.rlph import RLPH

NUM_EPISODES = 5000             # partite di training
GAME_LENGTH = 100               # turni massimi per partita
NUM_OPPONENTS = 2               # numero di opponents per ogni partita
EPSILON_START = 0.8             # esplorazione iniziale
EPSILON_END = 0.05              # esplorazione finale
QTABLE_PATH = "rl_qtable.json"  # salvato nella root del progetto
SAVE_EVERY = 50                 # salva ogni N episodi

EPSILON_DECAY = (EPSILON_START - EPSILON_END) / NUM_EPISODES

# Tutti i bot disponibili come avversari
ALL_BOTS = [
    # Stinky,
    Pixie,
    # Communist,
    Cluster,
    # Angry
]
BOT_WINS = [0] * len(ALL_BOTS)


def _find_map_csv() -> str:
    """Cerca automaticamente il CSV della mappa in data/."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidates = glob.glob(os.path.join(root, "data", "*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Nessun CSV trovato in data/. "
            "Imposta MAP_CSV manualmente."
        )
    return candidates[0]


def run_training():
    map_csv = _find_map_csv()
    print(f"[train_rl] Mappa: {map_csv}")

    agent = RLPH("rl", epsilon=EPSILON_START)
    wins = 0

    for ep in range(1, NUM_EPISODES + 1):

        # Decadimento lineare di epsilon
        agent.epsilon = max(EPSILON_END, EPSILON_START - EPSILON_DECAY * ep)

        # Reset della mappa ogni partita (gli owner vengono mutati in-place)
        game_map = Map.from_csv(map_csv)
        game = Game(game_map, game_length=GAME_LENGTH, enable_visualizer=False)

        # Sceglie 2 avversari casuali tra tutti i bot disponibili
        bot_classes = random.sample(ALL_BOTS, k=NUM_OPPONENTS)
        opponents = [
            cls(cls.__name__.replace("Player", "").lower())
            for cls in bot_classes
        ]

        game.add_player(agent)
        for opp in opponents:
            game.add_player(opp)

        print(game.get_alive_players())

        # Ordine di turno casuale
        # random.shuffle(game.players)
        game.get_game_state().set_players(game.players)   # type: ignore

        game.play(seed=None)

        # Reward terminale
        agent_won = False
        for p in game.get_alive_players():
            if p is agent:
                agent_won = True
                wins += 1
            else:
                for i, b in enumerate(ALL_BOTS):
                    if isinstance(p, b):
                        BOT_WINS[i] += 1

        agent.receive_terminal_reward(won=agent_won)

        # Log
        if ep % 10 == 0:
            print(f"Ep {ep:>4}/{NUM_EPISODES}  "
                  f"ε={agent.epsilon:.3f}  "
                  f"wins={wins}  "
                  f"other bots={
                      [
                        f"{c.__name__}: {BOT_WINS[i]} vittorie"
                        for i, c in enumerate(ALL_BOTS)
                      ]
                  }"
                  f"win-rate={wins/ep*100:.1f}%  "
                  f"stati Q={len(agent.Q)}")

        if ep % SAVE_EVERY == 0:
            agent.save(QTABLE_PATH)

    agent.save(QTABLE_PATH)
    print(f"\nTraining completo. Win-rate: {wins/NUM_EPISODES*100:.1f}%")


if __name__ == "__main__":
    run_training()
