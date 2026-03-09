from Risk.map import Country, Continent
from Risk.game_state import GameState


def get_most_contested_country(countries: list[Country], player) \
        -> Country | None:

    if len(countries) == 0:
        return None

    return max(
        countries,
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


def get_weakest_country(countries: list[Country]) -> Country | None:
    if len(countries) == 0:
        return None

    return min(
        countries,
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


def border_need_help(border: Country, border_army_size_min: int = 20) -> bool:
    return border.get_army_size() <= border_army_size_min


def continent_needs_help(
            player: object,
            continent: Continent,
            border_army_size_min: int = 20
        ) -> bool:
    if not continent.is_controlled_by(player):
        return True

    borders = continent.get_border_countries()

    if borders is None:
        return False

    for b in borders:
        if border_need_help(b, border_army_size_min):
            return True

    return False


def get_cluster_borders(root: list[Country] | Country) -> list[Country] | None:
    """
    Returns the border countries of a cluster — countries that have at least
    one enemy neighbor.

    A cluster is a group of countries controlled by the same player. Border
    countries are those on the edge of the cluster, meaning they are adjacent
    to at least one country owned by a different player.

    Args:
        root: A single Country or a list of Countries representing the cluster.

    Returns:
        A list of border Countries, or None if the input is None/empty or no
        border countries exist.
    """
    if root is None:
        return None

    cluster = root if isinstance(root, list) else [root]

    borders = [c for c in cluster if c.get_number_of_enemy_neighbors() > 0]

    return borders if borders else None
