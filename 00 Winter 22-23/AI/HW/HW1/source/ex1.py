# import math
# import random
import ast
import itertools
from typing import Tuple

import search
from utils import orientations, vector_add

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
            self.initial["taxis"][taxi_name]["max_fuel"] = self.initial["taxis"][taxi_name]["fuel"]
        for pass_name in self.initial["passengers"].keys():
            self.initial["passengers"][pass_name]["in_taxi"] = False
        self.initial["n_taxis"] = len(initial["taxis"])
        self.initial["n_passengers"] = len(initial["passengers"])
        # TODO: make sure we change these after actions / change these lines
        self.initial["n_unpicked"] = len(initial["passengers"])
        self.initial["n_picked_undelivered"] = 0
        self.initial["n_delivered"] = 0     # TODO: check extreme case where passenger already in his destination
        self.initial["map_size_height"] = len(self.initial["map"])
        self.initial["map_size_width"] = len(self.initial["map"][0])

        self.initial = dict_to_str(self.initial)

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
        "n_picked_undelivered": 0,
        "n_delivered": 0,
        "map_size_height": 4,
        "map_size_width": 4,
        }
        """

    def generate_locations(self, state: dict) -> dict:
        # get new locations by:
        # current location + one step in legal orientation (EAST, NORTH, WEST, SOUTH)
        possible_locations_by_taxi = dict()
        for taxi_name, taxi_dict in state["taxis"].items():
            curr_location = taxi_dict["location"]
            possible_locations = [
                vector_add(curr_location, orient) for orient in orientations
            ]
            possible_locations_by_taxi[taxi_name] = possible_locations

        return possible_locations_by_taxi

    def get_legal_moves_on_map(self, state: dict) -> dict:
        legal_locations_by_taxi = {}
        possible_locations_by_taxi = self.generate_locations(state)
        for taxi_name, taxi_dict in state["taxis"].items():
            # 1. check fuel > 0
            legal_locations = []
            if taxi_dict["fuel"] > 0:
                map_size_height = state["map_size_height"]
                map_size_width = state["map_size_width"]
                map_matrix = state["map"]

                possible_locations = possible_locations_by_taxi[taxi_name]
                for new_location in possible_locations:
                    x, y = new_location
                    # 2. check that the taxi doesn't get out of the map
                    # 3. check that the taxi is on a passable tile
                    if (
                        (0 <= x < map_size_width)
                        and (0 <= y < map_size_height)
                        and (map_matrix[x][y] != "I")
                    ):
                        legal_locations.append(new_location)
            legal_locations_by_taxi[taxi_name] = legal_locations  # TODO curr location?
        return legal_locations_by_taxi

    def get_legal_refuel(self, state: dict) -> dict:
        # Refueling can be performed only at gas stations
        legal_refuels_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            map_matrix = state["map"]
            x, y = taxi_dict["location"]  # current location of taxi
            # check that the location on map is "G"
            legal_refuel = (map_matrix[x][y] == "G")  # bool
            legal_refuels_by_taxi[taxi_name] = legal_refuel
        return legal_refuels_by_taxi

    def get_legal_pick_up(self, state: dict) -> dict:
        # Pick up passengers if they are on the same tile as the taxi.
        legal_pickups_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            capacity = taxi_dict["capacity"]
            passengers_in_taxi_list = taxi_dict["passengers_list"]
            legal_pickups = []
            # The number of passengers in the taxi has to be < taxi’s capacity.
            if len(passengers_in_taxi_list) < capacity:
                for passenger_name, passenger_dict in state["passengers"].items():
                    # check that location of taxi is the same as location of the passenger
                    if (taxi_dict["location"] == passenger_dict["location"]) & (
                        passenger_dict["location"] != passenger_dict["destination"]
                    ):
                        legal_pickups.append(passenger_name)
            legal_pickups_by_taxi[taxi_name] = legal_pickups
        return legal_pickups_by_taxi

    def get_legal_drop_off(self, state: dict) -> dict:
        # The passenger can only be dropped off on his destination tile
        # and will refuse to leave the vehicle otherwise.
        legal_drop_offs_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            legal_drop_offs = []
            # go over the passengers that's on the curr taxi
            for passenger_name in taxi_dict["passengers_list"]:
                passenger_dict = state["passengers"][passenger_name]
                # check that location of taxi is the same as destination of the passenger
                if taxi_dict["location"] == passenger_dict["destination"]:
                    legal_drop_offs.append(passenger_name)
            legal_drop_offs_by_taxi[taxi_name] = legal_drop_offs
        return legal_drop_offs_by_taxi

    def actions(self, state: str) -> Tuple[Tuple[Tuple]]:
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        state = str_to_dict(state)
        # -----------------------------------------------------------------
        # Atomic Actions: ["move", "pick_up", "drop_off", "refuel", "wait"]
        # explicit syntax:
        # (“move”, “taxi_name”, (x, y))
        # (“pick up”, “taxi_name”, “passenger_name”
        # (“drop off”, “taxi_name”, “passenger_name”)
        # ("refuel", "taxi_name")
        # ("wait", "taxi_name")

        # Full Action - a tuple with action for each taxi
        # Example: ((“move”, “taxi 1”, (1, 2)),
        #           (“wait”, “taxi 2”),
        #           (“pick up”, “very_fancy_taxi”, “Yossi”))
        # -----------------------------------------------------------------

        # For each taxi get Possible Atomic Actions

        legal_locations_by_taxi = self.get_legal_moves_on_map(
            state
        )  # DICT[taxi_name: list of (x,y) locations]
        legal_pickups_by_taxi = self.get_legal_pick_up(
            state
        )  # DICT[taxi_name: list of passengers names]
        legal_drop_offs_by_taxi = self.get_legal_drop_off(
            state
        )  # DICT[taxi_name: list of passengers names]
        legal_refuels_by_taxi = self.get_legal_refuel(
            state
        )  # DICT[taxi_name: True / False]

        # -----------------------------------------------------------------
        # Get Atomic Actions with right syntax
        atomic_actions_lists = []
        for taxi_name in state["taxis"].keys():
            atomic_actions = [("wait", taxi_name)]
            for location in legal_locations_by_taxi[taxi_name]:
                atomic_actions.append(("move", taxi_name, location))
            for passenger_name in legal_pickups_by_taxi[taxi_name]:
                atomic_actions.append(("pick up", taxi_name, passenger_name))
            for passenger_name in legal_drop_offs_by_taxi[taxi_name]:
                atomic_actions.append(("drop off", taxi_name, passenger_name))
            if legal_refuels_by_taxi[taxi_name]:
                atomic_actions.append(("refuel", taxi_name))
            atomic_actions_lists.append(atomic_actions)

        # -----------------------------------------------------------------
        # Get Actions - all permutations of atomic actions
        actions = list(itertools.product(*atomic_actions_lists))
        all_wait_action = tuple(
            [("wait", taxi_name) for taxi_name in state["taxis"].keys()]
        )
        assert all_wait_action in actions
        actions.remove(all_wait_action)

        # -----------------------------------------------------------------
        # For each action - Check That Taxis Don't Clash with each other
        #   == not going to the same location (therefore cannot pickup the same passenger)
        n_taxis = state["n_taxis"]
        if n_taxis > 1:
            legal_actions = []
            for action in actions:
                taxis_next_locations = []
                for atomic_action in action:        # TODO: NOTE changed from atomic_actions_lists to action
                    action_type = atomic_action[0]
                    taxi_name = atomic_action[1]
                    taxi_curr_location = state["taxis"][taxi_name]["location"]
                    if action_type == "move":
                        taxi_next_location = atomic_action[2]
                    else:
                        taxi_next_location = taxi_curr_location
                    taxis_next_locations.append(taxi_next_location)
                # check if there are 2 taxis in the same location
                legal_action = len(set(taxis_next_locations)) == n_taxis
                if legal_action:
                    legal_actions.append(action)
        else:  # n_taxis == 1 --> no clashing between taxis
            legal_actions = actions

        # -----------------------------------------------------------------
        # The result should be a tuple (or other iterable) of actions
        # as defined in the problem description file
        return tuple(legal_actions)

    def result(self, state: str, action: Tuple[Tuple]) -> str:
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        result_state = str_to_dict(state)

        for action_tuple in action:
            result_state = self._execute_action_tuple(result_state, action_tuple)

        # back into a hashable
        result_state = dict_to_str(result_state)
        return result_state

    def _execute_action_tuple(self, state: dict, action_tuple: Tuple) -> dict:
        """
        input: state dict, and an action tuple like: (“move”, “taxi_name”, (x, y))
        output: the state dict after preforming the action
        """
        actions_possible = MOVE, PICK_UP, DROP_OFF, REFUEL, WAIT = (
            "move",
            "pick up",
            "drop off",
            "refuel",
            "wait",
        )

        action_type = action_tuple[0]
        taxi_name = action_tuple[1]

        result_state = state.copy()

        # check input is legal
        assert (
            action_type in actions_possible
        ), f"{action_type} is not a possible action!"

        if action_type == MOVE:  # (“move”, “taxi_name”, (x, y))
            assert (
                len(action_tuple) == 3
            ), f"len of action_tuple should be 3: {action_tuple}"
            # taxi updates:
            #   fuel -= 1
            result_state["taxis"][taxi_name]["fuel"] -= 1
            #   location
            future_location = action_tuple[2]
            result_state["taxis"][taxi_name]["location"] = future_location

        elif action_type == PICK_UP:  # (“pick up”, “taxi_name”, “passenger_name”)
            assert (
                len(action_tuple) == 3
            ), f"len of action_tuple should be 3: {action_tuple}"
            passenger_name = action_tuple[2]

            # Taxi updates:
            #   taxi capacity -= 1
            result_state["taxis"][taxi_name]["capacity"] -= 1
            #   add passenger name to passengers_list of taxi
            result_state["taxis"][taxi_name]["passengers_list"].append(passenger_name)
            # Problem updates:
            #   n_picked_undelivered += 1
            result_state["n_picked_undelivered"] += 1
            #   n_unpicked -= 1
            result_state["n_unpicked"] -= 1
            # Passenger updates:
            #   update "in_taxi" of passenger to name of taxi
            result_state["passengers"][passenger_name]["in_taxi"] = taxi_name

        elif action_type == DROP_OFF:  # (“drop off”, “taxi_name”, “passenger_name”)
            assert (
                len(action_tuple) == 3
            ), f"len of action_tuple should be 3: {action_tuple}"
            passenger_name = action_tuple[2]
            # Taxi updates:
            #   taxi capacity += 1
            result_state["taxis"][taxi_name]["capacity"] += 1
            #   remove passenger name from passengers_list of taxi
            result_state["taxis"][taxi_name]["passengers_list"].remove(passenger_name)
            # Problem updates:
            #   n_picked_undelivered -= 1
            result_state["n_picked_undelivered"] -= 1
            #   n_delivered += 1
            result_state["n_delivered"] += 1
            # Passenger updates:
            #   passenger location = taxi location
            result_state["passengers"][passenger_name]["location"] = result_state["passengers"][passenger_name]['destination']
            #   update "in_taxi" of passenger to False
            result_state["passengers"][passenger_name]["in_taxi"] = False

        elif action_type == REFUEL:  # ("refuel", "taxi_name")
            assert (
                len(action_tuple) == 2
            ), f"len of action_tuple should be 2: {action_tuple}"
            # taxi updates:
            #   fuel = max_fuel
            result_state["taxis"][taxi_name]["fuel"] = result_state["taxis"][taxi_name][
                "max_fuel"
            ]

        elif action_type == WAIT:  # ("wait", "taxi_name")
            assert (
                len(action_tuple) == 2
            ), f"len of action_tuple should be 2: {action_tuple}"
            pass

        return result_state

    def goal_test(self, state):
        """Given a state, checks if this is the goal state.
        Returns True if it is, False otherwise."""
        state = str_to_dict(state)
        at_goal = True
        for passenger, params_dict in state["passengers"].items():
            location = params_dict["location"]
            dest = params_dict["destination"]
            if location != dest:
                at_goal = False
                break
        return at_goal

    def h(self, node):
        state = str_to_dict(node.state)
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
        state = str_to_dict(node.state)
        h_1 = (state["n_passengers"] * 2 + state["n_picked_undelivered"]) / state[
            "n_taxis"
        ]
        return h_1

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        state = str_to_dict(node.state)

        # D[i] = Manhattan distance between the initial location of an unpicked passenger i,
        # and her destination
        D = []
        # T[i] = Manhattan distance between the taxi where a picked but undelivered passenger is,
        # and her destination
        T = []

        for passenger, dict_params in node.state["passengers"].items():
            if not dict_params["in_taxi"]:  # then passenger is unpicked
                D.append(
                    manhattan_dist(dict_params["location"], dict_params["destination"])
                )
            else:  # then the passenger is picked
                taxi = node.state["taxis"][dict_params["in_taxi"]]
                T.append(manhattan_dist(taxi["location"], dict_params["destination"]))
        for passenger, dict_params in state["passengers"].items():
            if not dict_params["in_taxi"]:  # then passenger is unpicked
                D.append(
                    manhattan_dist(dict_params["location"], dict_params["destination"])
                )
            else:  # then the passenger is picked
                taxi = state["taxis"][dict_params["in_taxi"]]
                T.append(manhattan_dist(taxi["location"], dict_params["destination"]))

        value = (sum(D) + sum(T)) / state["n_taxis"]
        return value


def manhattan_dist(a, b):
    xA, yA = a
    xB, yB = b
    return abs(xA - xB) + abs(yA - yB)


def create_taxi_problem(game):
    return TaxiProblem(game)


def dict_to_str(d: dict) -> str:
    d_str = str(d)
    return d_str


def str_to_dict(s: str) -> dict:
    j_dict = ast.literal_eval(s)
    return j_dict
