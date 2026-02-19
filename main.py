from Risk.map import Map
from Risk.game import Game
from Risk.player import BlackPlayer

game_map = Map.from_csv('data/base_map.csv')

game = Game(game_map, 100)

p1 = BlackPlayer("black")
p2 = BlackPlayer("nero")
p3 = BlackPlayer("el primo")

game.set_players([p1, p2, p3])

game.play()
