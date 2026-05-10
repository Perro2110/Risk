from Risk.players.rlga import RLGA
from Risk.game import Game
from Risk.map import Map
from Risk.players.stinky import Stinky
from Risk.players.pixie import Pixie
from Risk.players.communist import Communist
from Risk.players.cluster import Cluster
from Risk.players.angry import Angry
from Risk.players.rlga import RLGA
import random
import os, sys
import pprint

NUM_OPPONENTS = 3
NUMBER_OF_GAMES = 1000
GAME_LENGTH = 100
SEED = 67

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

ALL_BOTS: list[type] = [
    Stinky,
    Pixie,
    Communist,
    Cluster,
    Angry,
]

def main(map_csv: str, n_games: int = NUMBER_OF_GAMES):
    """
    Gioca n_games partite per valutare i winrate e altre statistiche di ogni player.
    """
    random.seed(SEED)

    bot_stats = {
        cls.__name__.lower(): {
            # game actually played
            'games' : 0,

            # counters
            "wins" : 0,
            "rank_sum": 0.0,
            "terr_sum" : 0,
            "army_sum" : 0,
            "survival" : 0,

            # stats
            "win_rate" : 0.0,
            "rank_avg" : 0.0,
            "terr_avr" : 0.0,
            "army_avg" : 0.0,
            "survival_rate" : 0.0,
        } for cls in ALL_BOTS + [RLGA]
    }

    for i in range(n_games):
        print(f"game {i+1}")
        game_map = Map.from_csv(map_csv)
        game     = Game(game_map, game_length=GAME_LENGTH, enable_visualizer=False)

        agent    = RLGA("ga_agent")
        agent.load('models/ga_best.json')
        
        k = min(NUM_OPPONENTS, len(ALL_BOTS))
        bot_classes = random.sample(ALL_BOTS, k=k)
        opponents = [cls(cls.__name__.lower()) for cls in bot_classes]

        game.add_player(agent)
        for opp in opponents:
            game.add_player(opp)
        
        opponents.append(agent)

        # with contextlib.redirect_stdout(io.StringIO()), \
        #      contextlib.redirect_stderr(io.StringIO()):
        game.play()

        alive = game.get_alive_players()
        gm  = game.get_game_state().get_game_map()
        leaderboard = game.get_game_state().get_leaderboard()

        for p in opponents:
            bot_stats[p.__class__.__name__.lower()]['games'] += 1

            # winrate (more or less)
            if len(alive) == 1:
                bot_stats[p.__class__.__name__.lower()]['wins'] += alive[0] is p
            else:
                best_p = max(alive, key=lambda p: len(gm.get_owned_countries(p)))
                bot_stats[p.__class__.__name__.lower()]['wins'] += best_p is p
            
            bot_stats[p.__class__.__name__.lower()]['rank_sum'] += leaderboard.index(p) + 1
            bot_stats[p.__class__.__name__.lower()]['terr_sum'] += gm.get_num_countries(p)
            bot_stats[p.__class__.__name__.lower()]['army_sum'] += gm.get_owned_army_size(p) 
            
            bot_stats[p.__class__.__name__.lower()]['survival'] += p in alive

    for b in ALL_BOTS + [RLGA]:
        name = b.__name__.lower()
        g = bot_stats[name]['games'] or 1
        bot_stats[name]['win_rate']      = bot_stats[name]['wins']     / g
        bot_stats[name]['rank_avg']      = bot_stats[name]['rank_sum'] / g
        bot_stats[name]['terr_avr']      = bot_stats[name]['terr_sum'] / g
        bot_stats[name]['army_avg']      = bot_stats[name]['army_sum'] / g
        bot_stats[name]['survival_rate'] = bot_stats[name]['survival'] / g

    pprint.pp(bot_stats)

if __name__ == '__main__':
    main('data/base_map.csv')