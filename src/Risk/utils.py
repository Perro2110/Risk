from Risk.map import Country
from Risk.game import GameState


def get_most_contested_country(game_state: GameState, player: str) \
        -> Country | None:

    owned_countries = game_state.get_game_map() \
        .get_owned_countries(player)

    if len(owned_countries) == 0:
        return None

    return max(
        owned_countries,
        key=lambda country: country.get_number_of_enemy_neighbors(player)
    )


def get_most_contested_neighbour(country: Country, player: str = "") \
        -> Country | None:

    friendly_neighbours = country.get_friendly_neighbors(
        player if player else country.get_owner()
    )

    if len(friendly_neighbours) == 0:
        return None

    return max(
        friendly_neighbours,
        key=lambda country: country.get_number_of_enemy_neighbors(player)
    )


def weakest_enemy_neighbour(country: Country, player: str = "") \
        -> Country | None:

    enemy_neighbours = country.get_enemy_neighbors(
        player if player else country.get_owner()
    )

    if len(enemy_neighbours) == 0:
        return None

    return max(
        enemy_neighbours,
        key=lambda country: country.get_army_size()
    )
