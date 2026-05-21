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
import os
import glob


ALL_BOTS: list[type] = [
    Stinky,
    Pixie,
    Communist,
    Cluster,
    Angry,
]

if __name__ == '__main__':
    agent    = RLGA("ga_agent")
    agent.load('models/ga_best.json')
    bot_cls = random.choice(ALL_BOTS)
    opponent = bot_cls(bot_cls.__name__.lower())
    bot_cls2 = random.choice(ALL_BOTS)
    opponent2 = bot_cls2(bot_cls2.__name__.lower())
    bot_cls3 = random.choice(ALL_BOTS)
    opponent3 = bot_cls2(bot_cls3.__name__.lower())

    game_map = Map.from_csv('data/base_map.csv')
    game = Game(game_map, 100, True)
    game.add_player(agent)
    game.add_player(opponent)
    game.add_player(opponent2)
    game.add_player(opponent3)

    game.play(seed=None)