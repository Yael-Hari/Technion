# import math
# import random

import search

ids = ["316375872", "206014482"]


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        search.Problem.__init__(self, initial)
        for taxi_name in self.initial["taxis"].keys():
            self.initial["taxis"][taxi_name]["passengers_list"] = []
            # each taxi["fuel"] is the amount of fuel that the taxi starts with
            # and it is also the maximal amount of fuel
            self.initial["taxis"][taxi_name]["max_fuel"] = self.initial["taxis"][
                taxi_name
            ]["fuel"]
        for pass_name in self.initial["passengers"].keys():
            self.initial["passengers"][pass_name]["in_taxi"] = False
        self.initial["n_taxis"] = len(initial["taxis"])
        self.initial["n_passengers"] = len(initial["passengers"])
        # TODO: make sure we change these after actions / change these lines
        self.initial["n_unpicked"] = len(initial["passengers"])
        self.initial["n_picked_undelivered"] = 0
        self.initial["n_delivered"] = 0

        """
        State example
        {"map": [['P', 'P', 'P', 'P'],
                ['P', 'P', 'P', 'P'],
                ['P', 'I', 'G', 'P'],
                ['P', 'P', 'P', 'P'], ],
        "taxis": {'taxi 1': {"location": (3, 3),
                             "fuel": 15,
                             "max_fuel": 15,
                             "capacity": 2},
                             "passengers_list": []},
        "passengers": {'Yossi': {"location": (0, 0),
                                 "destination": (2, 3)},
                       'Moshe': {"location": (3, 1),
                                 "destination": (0, 0),
                                 "in_taxi": False}
                       },
        "n_taxis": 1,
        "n_passengers": 2,
        "n_unpicked": 2,
        "n_picked_undelivered": 0
        "n_delivered": 0
        }
        """

    def check_legal_move_on_map(self, state):
        
        legal_moves = []
        # 0. check that the move is one step in direction: left/ right/ up/ down
        # 1. check that the taxi don't get out of the map
        # 2. check that the taxi is on a passable tile
        # 3. check that there isn't other taxi on this tile
        # 4. check that there is fuel > 0
        # TODO: complete
        
        
        # TODO: decide where to update: 
        # fuel -= 1
        # location pn map

        return legal_moves
    
    def check_legal_refuel(self, state):
        # Refueling can be performed only at gas stations
        # check that the location on map is "G"
        # TODO: complete
        

        # TODO: decide where to update: fuel = max_fuel

    def check_legal_pick_up(self, state):
        # Pick up passengers if they are on the same tile as the taxi. 
        # check that location of taxi is the same as location of the passenger
        # TODO: complete
        
        # The number of passengers in the taxi at any given turn cannot exceed this taxi’s capacity.
        # check num_of_passengers_in_taxi < taxi_capcity
        # TODO: complete
        
        # TODO: decide where to update: 
        # Taxi updates:
        #   taxi capacity -= 1
        #   add passenger name from passengers_list of taxi
        # Problem updates:
        #   n_picked_undelivered += 1
        #   n_unpicked -= 1
        # Passenger updates:
        #   update "in_taxi" of passenger to name of taxi

    def check_legal_drop_off(self, state):
        # The passenger can only be dropped off on his destination tile and will refuse to leave the vehicle otherwise.
        # check that location of taxi is the same as destination of the passenger
        # TODO: complete
        
        # Drop off passengers on the same tile as the taxi
        # TODO: decide where to update: 
        # Taxi updates:
        #   taxi capacity += 1
        #   remove passenger name from passengers_list of taxi
        # Problem updates:
        #   n_picked_undelivered -= 1
        #   n_delivered += 1
        # Passenger updates:
        #   passenger location = taxi location
        #   update "in_taxi" of passenger to False

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        # TODO
        # Atomic Actions: ["move", "pick_up", "drop_off", "refuel", "wait"]
        # explivit syntax:
        # (“move”, “taxi_name”, (x, y))
        # (“pick up”, “taxi_name”, “passenger_name”
        # (“drop off”, “taxi_name”, “passenger_name”)
        # ("refuel", "taxi_name")
        # ("wait", "taxi_name")

        # Full Action - a tuple with action for each taxi
        # Example: ((“move”, “taxi 1”, (1, 2)),
        #           (“wait”, “taxi 2”),
        #           (“pick up”, “very_fancy_taxi”, “Yossi”))

        # for each taxi get possible atomic actions
        # TODO: complete
         
        # get all permutations of atomic actions
        # for each permutation check that the taxis don't clash
        #   not going to the same location (therfor cannot pickup the same passenger)
        # TODO: complete

    def result(self, state, action):
        # TODO
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""

    def goal_test(self, state):
        """Given a state, checks if this is the goal state.
        Returns True if it is, False otherwise."""
        at_goal = True
        for passenger, params_dict in self.initial["passengers"].items():
            location = params_dict["location"]
            dest = params_dict["destination"]
            if location != dest:
                at_goal = False
        return at_goal

    def h(self, node):
        # TODO
        """This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        return 0

    def h_1(self, node):
        """
        This is a simple heuristic
        (number of  passengers * 2 + the number of picked but yet undelivered passengers)
        /(number of taxis in the problem).
        """
        h_1 = (
            node.state["n_passengers"] * 2 + node.state["n_picked_undelivered"]
        ) / node.state["n_taxis"]
        return h_1

    def h_2(self, node):
        # TODO
        """
        This is a slightly more sophisticated Manhattan heuristic
        """

        """Feel free to add your own functions
        (-2, -2, None) means there was a timeout"""


def manhattan_dist(a, b):
    xA, yA = a
    xB, yB = b
    return abs(xA - xB) + abs(yA - yB)


def create_taxi_problem(game):
    return TaxiProblem(game)
