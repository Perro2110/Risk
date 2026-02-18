from Risk.map import Map
from Risk.game import Game
from Risk.player import BlackPlayer

game_map = Map.from_csv('data/base_map.csv')

game = Game(100, game_map)

p1 = BlackPlayer("b")
p2 = BlackPlayer("c")


game.set_players([p1,p2])

assert game_map is game.get_map()


game.play()

