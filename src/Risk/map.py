from __future__ import annotations
from multipledispatch import dispatch
import csv
# from Risk.player import object


class Country:
    """
    Risk game country class.
    owner is the player that controls the country, army_size is the number of
    armies in the country, neighbors is the list of neighboring countries.
    """

    def __init__(
                self,
                name: str,
                owner: object,
                army_size: int,
                neighbors: list[Country],
            ):
        self.owner = owner
        self.name = name
        self.army_size = army_size
        self.neighbors = neighbors

    # getter and setter for name
    def get_name(self) -> str:
        return self.name

    def set_name(self, name):
        self.name = name

    # getter and setter for owner
    def get_owner(self) -> object:
        return self.owner

    def set_owner(self, owner):
        self.owner = owner

    # getter and setter for army_size
    def get_army_size(self) -> int:
        return self.army_size

    def set_army_size(self, army_size):
        self.army_size = army_size

    # getter and setter for neighbors
    def get_neighbors(self) -> list[Country]:
        return self.neighbors

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    #####################################
    # utility functions
    #####################################
    def add_neighbor(self, neighbor):
        """ Adds a neighbor to the country """
        self.neighbors.append(neighbor)

    def is_neighbor(self, country) -> bool:
        """ Returns true if the country is a neighbor of the country """
        return country in self.neighbors

    def is_controlled_by(self, player: object) -> bool:
        """ Returns true if the country is controlled by the player """
        return self.owner == player

    def get_enemy_neighbors(self, player: object) -> list[Country]:
        """ Returns the list of enemy neighbors of the country """
        enemy_neighbors = []
        for neighbor in self.neighbors:
            if neighbor.get_owner() != player:
                enemy_neighbors.append(neighbor)
        return enemy_neighbors

    def get_number_of_enemy_neighbors(self, player: object | None = None) \
            -> int:
        """ Returns the number of enemy neighbors of the country """
        player = player if player else self.get_owner()
        return len(self.get_enemy_neighbors(player))

    def get_friendly_neighbors(self, player: object) -> list[Country]:
        """ Returns the list of friendly neighbors of the country """
        friendly_neighbors = []
        for neighbor in self.neighbors:
            if neighbor.get_owner() is player:
                friendly_neighbors.append(neighbor)
        return friendly_neighbors

    def get_number_of_friendly_neighbors(self, player: object) -> int:
        """ Returns the number of friendly neighbors of the country """
        return len(self.get_friendly_neighbors(player))

    def get_connected_friendly_countries(self) -> list[Country]:
        """
            Returns all friendly countries connected to this country
        """
        owner = self.get_owner()
        visited: set[Country] = set()
        queue: list[Country] = [self]

        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in current.get_friendly_neighbors(owner):
                if neighbor not in visited:
                    queue.append(neighbor)

        connected: list[Country] = [c for c in list(visited) if c is not self]
        return connected

    def __str__(self):
        return self.name


class Continent:
    """
    Risk game continent class.
    Vector of countries and reward for owning the continent (if you control
    all the countries in the continent).
    """

    def __init__(self, countries: list[Country], reward: int):
        self.countries = countries
        self.reward = reward

    # getter and setter for countries
    def get_countries(self) -> list[Country]:
        return self.countries

    def set_countries(self, countries: list[Country]):
        self.countries = countries

    # getter and setter for reward
    @dispatch()
    def get_reward(self):  # type: ignore (this removes warning)
        return self.reward

    def set_reward(self, reward):
        self.reward = reward

    #####################################
    # utility functions
    #####################################
    def is_controlled_by(self, player: object):
        """ Returns true if the continent is controlled by the player """
        for country in self.countries:
            if country.get_owner() is not player:
                return False
        return True

    @dispatch(object)
    def get_reward(self, player: object):
        """ Returns the reward for controlling the continent """
        if self.is_controlled_by(player):
            return self.reward
        return 0

    def get_best_cluster(self, player: object) -> list[Country] | None:
        countries = self.get_owned_countries(player)
        cluster_size = 0
        best_cluster = None

        while len(countries) > 0:
            current = countries[0]

            cluster = current.get_connected_friendly_countries() + [current]
            if len(cluster) > cluster_size:
                cluster_size = len(cluster)
                best_cluster = cluster

            # Remove the countries that are inside an already checked cluster
            countries = [c for c in countries if c not in cluster]

        return best_cluster

    def get_owned_countries(self, player: object) -> list[Country]:
        """
        Returns the list of countries in the continent that are controlled
        by the player.
        """
        owned_countries = []
        for country in self.countries:
            if country.get_owner() is player:
                owned_countries.append(country)
        return owned_countries

    def get_owned_army_size(self, player: object) -> int:
        """ Returns the army size on the map controlled by the player """
        countries = self.get_owned_countries(player)
        owned = 0
        for country in countries:
            owned += country.get_army_size()

        return owned


class Map:
    """ Risk game map class """

    def __init__(self, continents: list[Continent]):
        self.continents = continents

    # getter and setter for continents
    def get_continents(self) -> list[Continent]:
        return self.continents

    def set_continents(self, continents):
        self.continents = continents

    #####################################
    # utility functions
    #####################################
    def get_reward(self, player: object) -> int:
        """ Returns the total army reward for the player """
        reward = 0
        for continent in self.continents:
            reward += continent.get_reward(player)
        reward += self.get_num_countries(player) // 3
        return max(reward, 3)

    def get_countries(self) -> list[Country]:
        """ Returns all the countries """
        countries = []
        for continent in self.continents:
            for country in continent.get_countries():
                countries.append(country)
        return countries

    def get_num_countries(self, player: object) -> int:
        """ Returns the number of countries controlled by the player """
        num_countries = 0
        for continent in self.continents:
            for country in continent.get_countries():
                if country.get_owner() is player:
                    num_countries += 1
        return num_countries

    def get_owned_countries(self, player: object) -> list[Country]:
        """ Returns all countries on the map controlled by the player """
        owned = []
        for continent in self.continents:
            owned.extend(continent.get_owned_countries(player))
        return owned

    def get_owned_army_size(self, player: object) -> int:
        """ Returns the army size on the map controlled by the player """
        owned = 0
        for continent in self.continents:
            owned += continent.get_owned_army_size(player)
        return owned

    #####################################
    # class methods
    #####################################
    @classmethod
    def from_csv(cls, filepath: str, country_divisor: str = ';') -> Map:
        """
        Build a Map from a CSV file with columns:
            country, continent, continent_reward, neighbors

        The neighbors column is a semicolon-separated list of country names
        (e.g. "Alaska;Ontario;Greenland").
        """
        # create all Country objects and collect raw data
        countries: dict[str, Country] = {}
        rows: list[dict] = []

        with open(filepath, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row["country"].strip()
                country = Country(name, "", 0, [])
                countries[name] = country
                rows.append(row)

        # add neighbors
        for row in rows:
            name = row["country"].strip()
            neighbors = [
                n.strip() for n in row["neighbors"].split(country_divisor)
            ]
            countries[name].set_neighbors([countries[n] for n in neighbors])

        # create Continents
        continent_data: dict[str, dict] = {}  # name -> {reward, countries}
        for row in rows:
            name = row["continent"].strip()
            reward = int(row["continent_reward"].strip())
            if name not in continent_data:
                continent_data[name] = {"reward": reward, "countries": []}
            continent_data[name]["countries"].append(
                countries[row["country"].strip()]
            )

        continents = [
            Continent(data["countries"], data["reward"])
            for data in continent_data.values()
        ]

        # This creates and returns a Map with relative Contients and Countries
        return cls(continents)

    def __str__(self) -> str:
        return f'{
                [f'{c.get_name()} : {c.get_owner()} {c.get_army_size()} \n'
                    for c in self.get_countries()]
                }'
