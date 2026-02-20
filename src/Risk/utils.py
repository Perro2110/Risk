from Risk.map import Country
from Risk.game_state import GameState


def get_most_contested_country(game_state: GameState, player) \
        -> Country | None:

    owned_countries = game_state.get_game_map() \
        .get_owned_countries(player)

    if len(owned_countries) == 0:
        return None

    return max(
        owned_countries,
        key=lambda country: country.get_number_of_enemy_neighbors(player)
    )


def get_most_contested_neighbour(country: Country, player=None) \
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


def get_weakest_friendly_country(game_state: GameState, player) \
        -> Country | None:

    friendly_countries = game_state.get_game_map() \
        .get_owned_countries(player)

    if len(friendly_countries) == 0:
        return None

    return min(
        friendly_countries,
        key=lambda country: country.get_army_size()
    )


def weakest_enemy_neighbour(country: Country, player=None) \
        -> Country | None:

    enemy_neighbours = country.get_enemy_neighbors(
        player if player else country.get_owner()
    )

    if len(enemy_neighbours) == 0:
        return None

    return min(
        enemy_neighbours,
        key=lambda country: country.get_army_size()
    )


def weakest_enemy_neighbour_list(country: Country, player=None) \
        -> list[Country] | None:

    enemy_neighbours = country.get_enemy_neighbors(
        player if player else country.get_owner()
    )

    if len(enemy_neighbours) == 0:
        return None

    return sorted(
        enemy_neighbours,
        key=lambda country: country.get_army_size()
    )
