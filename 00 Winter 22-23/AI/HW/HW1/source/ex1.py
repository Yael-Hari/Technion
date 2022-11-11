import search
import random
import math


ids = ["316375872", "206014482"]


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        search.Problem.__init__(self, initial)
        for taxi_name in self.initial['taxis'].keys():
            self.initial['taxis'][taxi_name]['passengers_list'] = []

        """
        State example
        {"map": [['P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P'],
                    ['P', 'I', 'G', 'P'],
                    ['P', 'P', 'P', 'P'], ],
        "taxis": {'taxi 1': {"location": (3, 3),
                             "fuel": 15,
                             "capacity": 2},
                            "passengers_list": []},
        "passengers": {'Yossi': {"location": (0, 0),
                                 "destination": (2, 3)},
                       'Moshe': {"location": (3, 1),
                                 "destination": (0, 0)}
                       }}
        """

    def actions(self, state):
        # TODO
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""

    def result(self, state, action):
        # TODO
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""

    def goal_test(self, state):
        # TODO
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        at_goal = True
        for passenger, params_dict in self.initial['passengers'].items():
            location = params_dict['location']
            dest = params_dict['destination']
            if location != dest:
                at_goal = False
        return at_goal


    def h(self, node):
        # TODO
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        return 0

    def h_1(self, node):
        # TODO: check why ex
        """
        This is a simple heuristic
        (number of  passengers * 2 + the number of picked but yet undelivered passengers)
        /(number of taxis in the problem).
        """
        n_taxis = len(node.state["taxis"])
        n_passengers = len(node.state["passengers"])
        n_unpicked =
        n_picked_undelivered = 0


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

