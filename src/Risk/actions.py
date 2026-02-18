import random
from abc import ABC, abstractmethod
from Risk.map import Country


class Action(ABC):
    """ Action base class """

    @abstractmethod
    def execute(self):
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


class PlaceArmyAction(Action):
    """
    Place army action class.
        - country is the country where the armies will be placed
        - num_armies is the number of armies to place

    This is a first-phase action: the player chooses the number of armies to
    place and the country where to place them, then the action is executed
    and the armies are placed in the country.
    """

    def __init__(self, country: Country, num_armies: int = 0):
        self.country = country
        self.num_armies = num_armies

    def execute(self):
        """ Executes the action """
        self.country.set_army_size(
            self.country.get_army_size() + self.num_armies
        )


class FortifyAction(Action):
    """ Move army action class """

    def __init__(
                self,
                from_country: Country,
                to_country: Country,
                num_armies: int = 0
            ):
        self.from_country = from_country
        self.to_country = to_country
        self.num_armies = num_armies

    def execute(self):
        """ Executes the action """
        self.from_country.set_army_size(
            self.from_country.get_army_size() - self.num_armies)
        self.to_country.set_army_size(
            self.to_country.get_army_size() + self.num_armies)

class AttackAction(Action):
    """
    Attack action class.
        - from_country is the country from which the attack will be launched
        - to_country is the country that will be attacked
        - num_armies is the number of armies to attack with (must be less
                     than the number of armies in the from_country)

    This is a second-phase action: the player chooses the country to attack
    from, the country to attack, and the number of armies to attack with,
    then the action is executed and the battle is resolved according to the
    rules of Risk.
    """

    def __init__(
                self,
                from_country: Country,
                to_country: Country,
                num_armies: int = 0,
                army_want_to_move: int = 0,
            ):
        self.army_want_to_move = army_want_to_move
        self.from_country = from_country
        self.to_country = to_country
        self.num_armies = num_armies

    def execute(self):
        """ Executes the action """
        num_attackers = self.num_armies

        # Notice: we assume the defenders always roll the maximum number of
        # dice (3 if they have 3+ armies, 2 if they have 2, 1 if they have 1).
        num_defenders = min(3, self.to_country.get_army_size())

        # Classic Risk attack rules — roll dice for attackers and defenders
        attack_rolls = sorted(
            [random.randint(1, 6) for _ in range(num_attackers)],
            reverse=True
        )
        defence_rolls = sorted(
            [random.randint(1, 6) for _ in range(num_defenders)],
            reverse=True
        )

        # Compare rolls and determine outcome
        for attacker_roll, defender_roll in zip(attack_rolls, defence_rolls):
            if attacker_roll > defender_roll:
                # Attacker wins, defender loses an army
                self.to_country.set_army_size(
                    self.to_country.get_army_size() - 1)
            else:
                # Defender wins, attacker loses an army
                self.from_country.set_army_size(
                    self.from_country.get_army_size() - 1)

        # If the attacker conquered the to_country, move armies in
        if self.to_country.get_army_size() == 0:
            fortify_action = FortifyAction(
                self.from_country,
                self.to_country,
                # we move army want to move and num army survived 
                (self.army_want_to_move + self.num_armies) 
            )
            fortify_action.execute()
