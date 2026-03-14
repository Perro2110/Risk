"""
train_rl.py — Training loop per RLPlayer.

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
from Risk.player import BlackPlayer, GreenPlayer, PurplePlayer, RedPlayer, YellowPlayer
from Risk.rl_player import RLPlayer

# ─── Configurazione ───────────────────────────────────────────────────────────
NUM_EPISODES  = 500    # partite di training
GAME_LENGTH   = 100    # turni massimi per partita
EPSILON_START = 0.5    # esplorazione iniziale
EPSILON_END   = 0.05   # esplorazione finale
QTABLE_PATH   = "rl_qtable.json"  # salvato nella root del progetto
SAVE_EVERY    = 50     # salva ogni N episodi
ENABLE_VIZ    = False  # True per guardare le partite (molto lento)
# ──────────────────────────────────────────────────────────────────────────────

EPSILON_DECAY = (EPSILON_START - EPSILON_END) / NUM_EPISODES

# Tutti i bot disponibili come avversari
ALL_BOTS = [RedPlayer, BlackPlayer, YellowPlayer, PurplePlayer, GreenPlayer]


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

    agent = RLPlayer("rl", epsilon=EPSILON_START)
    wins  = 0

    for episode in range(1, NUM_EPISODES + 1):

        # Decadimento lineare di epsilon
        agent.epsilon = max(EPSILON_END, EPSILON_START - EPSILON_DECAY * episode)

        # Ricostruisce la mappa ogni partita (gli owner vengono mutati in-place)
        game_map = Map.from_csv(map_csv)
        game     = Game(game_map, game_length=GAME_LENGTH, enable_visualizer=ENABLE_VIZ)

        # Sceglie 2 avversari casuali tra tutti i bot disponibili
        bot_classes = random.sample(ALL_BOTS, k=2)
        opponents   = [cls(cls.__name__.replace("Player", "").lower())
                       for cls in bot_classes]

        game.add_player(agent)
        for opp in opponents:
            game.add_player(opp)

        # Ordine di turno casuale
        random.shuffle(game.players)
        game.get_game_state().set_players(game.players)

        game.play(seed=None)

        # Reward terminale
        alive     = game.get_alive_players()
        agent_won = (agent in alive and len(alive) == 1)
        if agent_won:
            wins += 1
        agent.receive_terminal_reward(won=agent_won)

        # Log
        if episode % 10 == 0:
            print(f"Ep {episode:>4}/{NUM_EPISODES}  "
                  f"ε={agent.epsilon:.3f}  "
                  f"wins={wins}  "
                  f"win-rate={wins/episode*100:.1f}%  "
                  f"stati Q={len(agent.Q)}")

        if episode % SAVE_EVERY == 0:
            agent.save(QTABLE_PATH)

    agent.save(QTABLE_PATH)
    print(f"\nTraining completo. Win-rate: {wins/NUM_EPISODES*100:.1f}%")


if __name__ == "__main__":
    run_training()