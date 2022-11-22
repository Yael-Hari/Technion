import time
from dataclasses import dataclass, field

import ex1
import search


def timeout_exec(func, args=(), kwargs={}, timeout_duration=10, default=None):
    """This function will spawn a thread and run the given function
    using the args, kwargs and return the given default value if the
    timeout_duration is exceeded.
    """
    import threading

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default

        def run(self):
            # try:
            self.result = func(*args, **kwargs)
            # except Exception as e:
            #    self.result = (-3, -3, e)

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.is_alive():
        return default
    else:
        return it.result


def check_problem(p, search_method, timeout):
    """Constructs a problem using ex1.create_wumpus_problem,
    and solves it using the given search_method with the given timeout.
    Returns a tuple of (solution length, solution time, solution)"""

    """ (-2, -2, None) means there was a timeout
    (-3, -3, ERR) means there was some error ERR during search """

    t1 = time.time()
    s = timeout_exec(search_method, args=[p], timeout_duration=timeout)
    t2 = time.time()

    if isinstance(s, search.Node):
        solve = s
        solution = list(map(lambda n: n.action, solve.path()))[1:]
        return (len(solution), t2 - t1, solution)
    elif s is None:
        return (-2, -2, None)
    else:
        return s


def solve_problems(problems):
    solved = 0
    for problem in problems:
        try:
            p = ex1.create_taxi_problem(problem)
        except Exception as e:
            print("Error creating problem: ", e)
            return None
        timeout = 60
        result = check_problem(p, (lambda p: search.astar_search(p, p.h)), timeout)
        print("A*: ", result)
        if result[2] != None:
            if result[0] != -3:
                solved = solved + 1


def main():
    print(ex1.ids)
    """Here goes the input you want to check"""
    problems = [
        {
            "map": [
                ["P", "P", "P", "P"],
                ["P", "P", "P", "P"],
                ["P", "I", "G", "P"],
                ["P", "P", "P", "P"],
            ],
            "taxis": {"taxi 1": {"location": (3, 3), "fuel": 15, "capacity": 2}},
            "passengers": {
                "Yossi": {"location": (0, 0), "destination": (2, 3)},
                "Moshe": {"location": (3, 1), "destination": (0, 0)},
            },
        },
        {
            "map": [
                ["P", "P", "I", "P"],
                ["P", "P", "P", "P"],
                ["P", "I", "G", "P"],
                ["P", "P", "P", "P"],
            ],
            "taxis": {"taxi 1": {"location": (3, 3), "fuel": 15, "capacity": 2}},
            "passengers": {
                "Dana": {"location": (0, 0), "destination": (2, 3)},
                "Yael": {"location": (3, 1), "destination": (0, 0)},
            },
        },
    ]
    ten = [
        {
            'map': [['P', 'P', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P']],
            'taxis': {'taxi 1': {'location': (0, 2), 'fuel': 5, 'capacity': 3}},
            'passengers': {'Omar': {'location': (0, 0), 'destination': (1, 1)},
                        'Omer': {'location': (2, 1), 'destination': (1, 2)},
                        'Daniel': {'location': (0, 2), 'destination': (0, 1)}},
        },
        {
            'map': [['P', 'P', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P']],
            'taxis': {'taxi 1': {'location': (2, 0), 'fuel': 5, 'capacity': 2},
                    'taxi 2': {'location': (0, 1), 'fuel': 6, 'capacity': 2}},
            'passengers': {'Iris': {'location': (0, 0), 'destination': (1, 4)},
                        'Daniel': {'location': (3, 1), 'destination': (2, 1)},
                        'Freyja': {'location': (2, 3), 'destination': (2, 4)},
                        'Tamar': {'location': (3, 0), 'destination': (3, 2)}},
        },
    ]
    additional_inputs = [
        {
            'map': [['P', 'P', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P']],
            'taxis': {'taxi 1': {'location': (1, 3), 'fuel': 10, 'capacity': 3}},
            'passengers': {'Michael': {'location': (3, 4), 'destination': (2, 1)},
                        'Freyja': {'location': (0, 0), 'destination': (2, 1)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P']],
            'taxis': {'taxi 1': {'location': (0, 2), 'fuel': 5, 'capacity': 3}},
            'passengers': {'Omar': {'location': (0, 0), 'destination': (1, 1)},
                        'Omer': {'location': (2, 1), 'destination': (1, 2)},
                        'Daniel': {'location': (0, 2), 'destination': (0, 1)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P']],
            'taxis': {'taxi 1': {'location': (0, 2), 'fuel': 8, 'capacity': 2}},
            'passengers': {'Eitan': {'location': (1, 0), 'destination': (0, 2)},
                        'Omer': {'location': (3, 4), 'destination': (0, 2)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'P', 'P'],
                    ['P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P'], ['P', 'P', 'P', 'G', 'P']],
            'taxis': {'taxi 1': {'location': (1, 2), 'fuel': 18, 'capacity': 1}},
            'passengers': {'Freyja': {'location': (2, 0), 'destination': (4, 2)},
                        'Wolfgang': {'location': (2, 1), 'destination': (1, 4)},
                        'Jacob': {'location': (3, 4), 'destination': (3, 2)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'P', 'P'],
                    ['P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'P', 'G', 'P']],
            'taxis': {'taxi 1': {'location': (0, 2), 'fuel': 15, 'capacity': 2}},
            'passengers': {'Omar': {'location': (0, 3), 'destination': (3, 2)},
                        'John': {'location': (1, 0), 'destination': (4, 0)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'P', 'P'],
                    ['P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'P', 'G', 'P']],
            'taxis': {'taxi 1': {'location': (1, 0), 'fuel': 15, 'capacity': 3}},
            'passengers': {'Moshe': {'location': (4, 2), 'destination': (0, 1)},
                        'Freyja': {'location': (2, 3), 'destination': (0, 3)}},
        },

        {
            'map': [['P', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'P', 'P', 'P', 'I'],
                    ['P', 'P', 'I', 'P', 'P', 'I', 'P'],
                    ['P', 'G', 'P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'G', 'I', 'P', 'P', 'P']],
            'taxis': {'taxi 1': {'location': (5, 6), 'fuel': 21, 'capacity': 3}},
            'passengers': {'Omer': {'location': (1, 5), 'destination': (2, 2)},
                        'Roee': {'location': (2, 1), 'destination': (4, 3)},
                        'Dana': {'location': (4, 2), 'destination': (5, 2)},
                        'Efrat': {'location': (5, 6), 'destination': (2, 3)}},
        },

        {
            'map': [['P', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'P', 'P', 'P', 'I'],
                    ['P', 'P', 'I', 'P', 'P', 'I', 'P'],
                    ['P', 'G', 'P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'G', 'I', 'P', 'P', 'P']],
            'taxis': {'taxi 1': {'location': (5, 5), 'fuel': 6, 'capacity': 2}},
            'passengers': {'Janet': {'location': (5, 4), 'destination': (1, 4)},
                        'Omer': {'location': (1, 5), 'destination': (5, 0)},
                        'Oliver': {'location': (4, 4), 'destination': (3, 4)}},
        },

        {
            'map': [['P', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'P', 'P', 'P', 'I'],
                    ['P', 'P', 'I', 'P', 'P', 'I', 'P'],
                    ['P', 'G', 'P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'G', 'I', 'P', 'P', 'P']],
            'taxis': {'taxi 1': {'location': (1, 0), 'fuel': 6, 'capacity': 2}},
            'passengers': {'Yael': {'location': (5, 4), 'destination': (1, 6)},
                        'Janet': {'location': (5, 5), 'destination': (3, 6)},
                        'Francois': {'location': (5, 0), 'destination': (4, 6)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P']],
            'taxis': {'taxi 1': {'location': (2, 0), 'fuel': 5, 'capacity': 2},
                    'taxi 2': {'location': (0, 1), 'fuel': 6, 'capacity': 2}},
            'passengers': {'Iris': {'location': (0, 0), 'destination': (1, 4)},
                        'Daniel': {'location': (3, 1), 'destination': (2, 1)},
                        'Freyja': {'location': (2, 3), 'destination': (2, 4)},
                        'Tamar': {'location': (3, 0), 'destination': (3, 2)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'P', 'P'],
                    ['P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'P', 'G', 'P']],
            'taxis': {'taxi 1': {'location': (4, 4), 'fuel': 17, 'capacity': 1},
                    'taxi 2': {'location': (2, 1), 'fuel': 18, 'capacity': 3}},
            'passengers': {'Freyja': {'location': (4, 2), 'destination': (4, 4)},
                        'Dave': {'location': (0, 0), 'destination': (3, 3)},
                        'Janet': {'location': (1, 4), 'destination': (3, 3)},
                        'Francois': {'location': (3, 1), 'destination': (0, 3)}},
        },

        {
            'map': [['P', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'P', 'P', 'P', 'I'],
                    ['P', 'P', 'I', 'P', 'P', 'I', 'P'],
                    ['P', 'G', 'P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'G', 'I', 'P', 'P', 'P']],
            'taxis': {'taxi 1': {'location': (4, 0), 'fuel': 20, 'capacity': 2},
                    'taxi 2': {'location': (4, 3), 'fuel': 10, 'capacity': 3}},
            'passengers': {'Dana': {'location': (2, 6), 'destination': (4, 2)},
                        'Uri': {'location': (3, 5), 'destination': (1, 4)},
                        'Ali': {'location': (1, 2), 'destination': (1, 3)},
                        'Daniel': {'location': (1, 3), 'destination': (3, 4)},
                        'Wolfgang': {'location': (3, 1), 'destination': (3, 6)},
                        'Noa': {'location': (2, 4), 'destination': (3, 6)},
                        'Ayelet': {'location': (1, 2), 'destination': (0, 4)},
                        'Khaled': {'location': (5, 1), 'destination': (3, 6)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'G', 'I', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'I', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'I', 'I', 'P', 'I', 'G'],
                    ['P', 'I', 'P', 'P', 'P', 'I', 'P'],
                    ['G', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'P', 'I', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'G', 'I', 'P', 'P']],
            'taxis': {'taxi 1': {'location': (8, 6), 'fuel': 25, 'capacity': 2},
                    'taxi 2': {'location': (5, 4), 'fuel': 28, 'capacity': 1}},
            'passengers': {'Michael': {'location': (0, 1), 'destination': (7, 6)},
                        'John': {'location': (9, 1), 'destination': (4, 6)},
                        'Lian': {'location': (7, 2), 'destination': (7, 0)},
                        'Francois': {'location': (0, 4), 'destination': (6, 2)},
                        'Tamar': {'location': (0, 4), 'destination': (8, 0)},
                        'Emma': {'location': (8, 6), 'destination': (3, 2)},
                        'Freyja': {'location': (5, 5), 'destination': (9, 6)},
                        'Gal': {'location': (7, 2), 'destination': (9, 2)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'G', 'I', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'I', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'I', 'I', 'P', 'I', 'G'],
                    ['P', 'I', 'P', 'P', 'P', 'I', 'P'],
                    ['G', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'P', 'I', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'G', 'I', 'P', 'P']],
            'taxis': {'taxi 1': {'location': (6, 4), 'fuel': 20, 'capacity': 1},
                    'taxi 2': {'location': (4, 3), 'fuel': 29, 'capacity': 2},
                    'taxi 3': {'location': (2, 6), 'fuel': 24, 'capacity': 3}},
            'passengers': {'Roee': {'location': (1, 0), 'destination': (3, 0)},
                        'Freyja': {'location': (0, 4), 'destination': (0, 0)},
                        'Moshe': {'location': (2, 4), 'destination': (5, 4)},
                        'Iris': {'location': (1, 6), 'destination': (5, 3)},
                        'Sergei': {'location': (9, 0), 'destination': (0, 5)},
                        'Jacob': {'location': (3, 4), 'destination': (4, 5)},
                        'Daniel': {'location': (8, 2), 'destination': (9, 1)},
                        'Lian': {'location': (8, 3), 'destination': (6, 3)},
                        'Yael': {'location': (3, 1), 'destination': (2, 4)},
                        'Greta': {'location': (7, 5), 'destination': (5, 1)},
                        'Reema': {'location': (3, 6), 'destination': (9, 3)},
                        'Emma': {'location': (5, 4), 'destination': (3, 0)},
                        'Khaled': {'location': (8, 5), 'destination': (3, 0)},
                        'Jana': {'location': (4, 6), 'destination': (2, 4)},
                        'Wolfgang': {'location': (8, 2), 'destination': (3, 0)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'G', 'I', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'I', 'P', 'P', 'I', 'P', 'I', 'P', 'I'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'G', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P'],
                    ['G', 'P', 'P', 'I', 'P', 'P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'I', 'I'],
                    ['P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'P', 'G', 'I', 'P', 'P', 'P', 'P', 'I']],
            'taxis': {'taxi 1': {'location': (0, 3), 'fuel': 17, 'capacity': 3},
                    'taxi 2': {'location': (0, 0), 'fuel': 12, 'capacity': 2}},
            'passengers': {'Francois': {'location': (6, 6), 'destination': (0, 4)},
                        'Wolfgang': {'location': (2, 1), 'destination': (4, 0)},
                        'Amir': {'location': (1, 7), 'destination': (4, 0)},
                        'Freyja': {'location': (9, 5), 'destination': (1, 7)},
                        'Janet': {'location': (1, 0), 'destination': (7, 8)},
                        'Khaled': {'location': (3, 9), 'destination': (1, 9)},
                        'Michael': {'location': (3, 0), 'destination': (5, 6)},
                        'Meytal': {'location': (1, 1), 'destination': (1, 6)},
                        'Dave': {'location': (7, 5), 'destination': (7, 9)},
                        'Lian': {'location': (7, 7), 'destination': (8, 2)},
                        'Efrat': {'location': (0, 2), 'destination': (1, 5)},
                        'Gal': {'location': (8, 7), 'destination': (1, 4)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'G', 'I', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'I', 'P'],
                    ['P', 'P', 'I', 'P', 'P', 'I', 'P', 'I', 'P', 'I'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'G', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P'],
                    ['G', 'P', 'P', 'I', 'P', 'P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'I', 'I'],
                    ['P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'P', 'G', 'I', 'P', 'P', 'P', 'P', 'I']],
            'taxis': {'taxi 1': {'location': (4, 0), 'fuel': 14, 'capacity': 1},
                    'taxi 2': {'location': (1, 9), 'fuel': 21, 'capacity': 1},
                    'taxi 3': {'location': (8, 3), 'fuel': 38, 'capacity': 2}},
            'passengers': {'Omer': {'location': (4, 8), 'destination': (1, 9)},
                        'Gal': {'location': (2, 3), 'destination': (8, 1)},
                        'Jana': {'location': (9, 6), 'destination': (5, 4)},
                        'Reema': {'location': (7, 0), 'destination': (5, 0)},
                        'Dana': {'location': (4, 7), 'destination': (2, 5)},
                        'Kobi': {'location': (0, 6), 'destination': (6, 2)},
                        'Ayelet': {'location': (5, 4), 'destination': (7, 3)},
                        'Meytal': {'location': (9, 8), 'destination': (7, 5)},
                        'Oliver': {'location': (4, 9), 'destination': (5, 6)},
                        'Amir': {'location': (7, 3), 'destination': (4, 1)},
                        'Sergei': {'location': (2, 4), 'destination': (5, 4)},
                        'Daniel': {'location': (1, 0), 'destination': (0, 5)},
                        'Francois': {'location': (3, 3), 'destination': (3, 9)},
                        'Yael': {'location': (2, 1), 'destination': (7, 7)},
                        'Tamar': {'location': (5, 0), 'destination': (9, 0)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'I', 'P', 'I', 'P', 'P', 'P', 'P', 'G', 'P', 'P', 'P'],
                    ['P', 'P', 'G', 'P', 'I', 'P', 'P', 'I', 'P', 'I', 'I', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'I', 'P'],
                    ['P', 'I', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'I', 'P', 'I', 'G', 'P', 'I', 'P', 'P', 'P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'G', 'I', 'P', 'P', 'P', 'P', 'P', 'I'],
                    ['P', 'I', 'I', 'I', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'G', 'P', 'I', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P']],
            'taxis': {'taxi 1': {'location': (7, 11), 'fuel': 60, 'capacity': 3},
                    'taxi 2': {'location': (11, 4), 'fuel': 49, 'capacity': 1}},
            'passengers': {'Omer': {'location': (13, 6), 'destination': (4, 2)},
                        'Michael': {'location': (8, 7), 'destination': (2, 3)},
                        'Francois': {'location': (11, 4), 'destination': (4, 13)},
                        'Jacob': {'location': (9, 10), 'destination': (11, 8)},
                        'Kobi': {'location': (11, 7), 'destination': (9, 13)},
                        'Mohammad': {'location': (1, 0), 'destination': (6, 5)},
                        'Noa': {'location': (5, 2), 'destination': (6, 1)},
                        'Ali': {'location': (3, 11), 'destination': (10, 0)},
                        'Irina': {'location': (10, 5), 'destination': (9, 9)},
                        'Jana': {'location': (7, 11), 'destination': (8, 4)},
                        'Avi': {'location': (2, 0), 'destination': (5, 11)},
                        'Roee': {'location': (6, 7), 'destination': (11, 1)},
                        'Dave': {'location': (6, 1), 'destination': (7, 3)},
                        'Daniel': {'location': (8, 9), 'destination': (3, 8)},
                        'Gal': {'location': (0, 0), 'destination': (12, 12)},
                        'Khaled': {'location': (1, 0), 'destination': (0, 5)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'I', 'P', 'I', 'P', 'P', 'P', 'P', 'G', 'P', 'P', 'P'],
                    ['P', 'P', 'G', 'P', 'I', 'P', 'P', 'I', 'P', 'I', 'I', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'I', 'P'],
                    ['P', 'I', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'I', 'P', 'I', 'G', 'P', 'I', 'P', 'P', 'P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'G', 'I', 'P', 'P', 'P', 'P', 'P', 'I'],
                    ['P', 'I', 'I', 'I', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'G', 'P', 'I', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P']],
            'taxis': {'taxi 1': {'location': (5, 9), 'fuel': 32, 'capacity': 1},
                    'taxi 2': {'location': (4, 11), 'fuel': 56, 'capacity': 2},
                    'taxi 3': {'location': (10, 0), 'fuel': 24, 'capacity': 3}},
            'passengers': {'Greta': {'location': (0, 12), 'destination': (6, 6)},
                        'Moshe': {'location': (11, 10), 'destination': (3, 5)},
                        'Nour': {'location': (12, 8), 'destination': (12, 13)},
                        'Tamar': {'location': (8, 7), 'destination': (2, 9)},
                        'Amir': {'location': (0, 14), 'destination': (2, 9)},
                        'Noa': {'location': (3, 2), 'destination': (11, 9)},
                        'Omer': {'location': (2, 11), 'destination': (1, 7)},
                        'Omar': {'location': (11, 0), 'destination': (7, 2)},
                        'Wolfgang': {'location': (2, 8), 'destination': (10, 12)},
                        'Kobi': {'location': (11, 4), 'destination': (8, 14)},
                        'Meytal': {'location': (11, 9), 'destination': (9, 4)},
                        'Jacob': {'location': (6, 2), 'destination': (12, 9)},
                        'Iris': {'location': (9, 1), 'destination': (5, 1)},
                        'Mohammad': {'location': (13, 14), 'destination': (14, 13)},
                        'Ayelet': {'location': (10, 0), 'destination': (11, 3)},
                        'Roee': {'location': (4, 8), 'destination': (1, 7)},
                        'Gal': {'location': (9, 10), 'destination': (7, 8)},
                        'Jana': {'location': (1, 7), 'destination': (8, 13)},
                        'Francois': {'location': (8, 3), 'destination': (7, 11)},
                        'John': {'location': (12, 6), 'destination': (1, 0)},
                        'Lian': {'location': (9, 11), 'destination': (1, 9)}},
        },

        {
            'map': [['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'I', 'P', 'I', 'P', 'P', 'P', 'P', 'G', 'P', 'P', 'P'],
                    ['P', 'P', 'G', 'P', 'I', 'P', 'P', 'I', 'P', 'I', 'I', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'I', 'P'],
                    ['P', 'I', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'I', 'P', 'I', 'G', 'P', 'I', 'P', 'P', 'P', 'P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'G', 'I', 'P', 'P', 'P', 'P', 'P', 'I'],
                    ['P', 'I', 'I', 'I', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'G', 'P', 'I', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'G', 'P'],
                    ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P']],
            'taxis': {'taxi 1': {'location': (0, 7), 'fuel': 37, 'capacity': 1},
                    'taxi 2': {'location': (7, 3), 'fuel': 42, 'capacity': 1},
                    'taxi 3': {'location': (4, 10), 'fuel': 52, 'capacity': 2},
                    'taxi 4': {'location': (7, 6), 'fuel': 30, 'capacity': 1},
                    'taxi 5': {'location': (11, 9), 'fuel': 18, 'capacity': 1},
                    'taxi 6': {'location': (4, 2), 'fuel': 38, 'capacity': 1},
                    'taxi 7': {'location': (3, 9), 'fuel': 22, 'capacity': 3},
                    'taxi 8': {'location': (5, 5), 'fuel': 17, 'capacity': 1}},
            'passengers': {'Yael': {'location': (14, 2), 'destination': (2, 0)},
                        'Irina': {'location': (5, 9), 'destination': (7, 6)},
                        'Dave': {'location': (1, 14), 'destination': (6, 6)},
                        'Uri': {'location': (6, 12), 'destination': (13, 14)},
                        'Iris': {'location': (12, 9), 'destination': (7, 8)},
                        'Mohammad': {'location': (11, 5), 'destination': (0, 12)},
                        'John': {'location': (6, 3), 'destination': (11, 14)},
                        'Greta': {'location': (0, 11), 'destination': (4, 13)},
                        'Yossi': {'location': (0, 14), 'destination': (3, 0)},
                        'Amir': {'location': (2, 0), 'destination': (12, 7)},
                        'Dana': {'location': (10, 6), 'destination': (9, 1)},
                        'Tamar': {'location': (9, 0), 'destination': (2, 9)},
                        'Ali': {'location': (7, 5), 'destination': (2, 12)},
                        'Reema': {'location': (13, 0), 'destination': (2, 2)},
                        'Eitan': {'location': (7, 1), 'destination': (4, 2)},
                        'Kobi': {'location': (2, 3), 'destination': (9, 8)},
                        'Janet': {'location': (0, 6), 'destination': (6, 9)},
                        'Meytal': {'location': (9, 3), 'destination': (3, 0)},
                        'Omer': {'location': (10, 13), 'destination': (5, 2)},
                        'Sergei': {'location': (7, 1), 'destination': (11, 9)},
                        'Wolfgang': {'location': (13, 8), 'destination': (1, 14)},
                        'Noa': {'location': (11, 5), 'destination': (14, 2)},
                        'Roee': {'location': (12, 7), 'destination': (4, 11)},
                        'Ayelet': {'location': (12, 13), 'destination': (1, 9)},
                        'Khaled': {'location': (13, 12), 'destination': (11, 13)},
                        'Moshe': {'location': (5, 0), 'destination': (2, 13)},
                        'Omar': {'location': (11, 13), 'destination': (8, 5)},
                        'Daniel': {'location': (9, 13), 'destination': (7, 10)},
                        'Lian': {'location': (13, 8), 'destination': (2, 3)},
                        'Efrat': {'location': (12, 4), 'destination': (6, 2)},
                        'Francois': {'location': (11, 1), 'destination': (5, 0)},
                        'Jana': {'location': (12, 7), 'destination': (10, 5)},
                        'Jacob': {'location': (7, 14), 'destination': (13, 1)},
                        'Avi': {'location': (11, 0), 'destination': (9, 14)},
                        'Nour': {'location': (3, 12), 'destination': (6, 7)},
                        'Freyja': {'location': (7, 14), 'destination': (2, 14)},
                        'Michael': {'location': (0, 7), 'destination': (5, 10)},
                        'Gal': {'location': (7, 9), 'destination': (4, 10)},
                        'Oliver': {'location': (2, 13), 'destination': (2, 0)},
                        'Emma': {'location': (4, 8), 'destination': (11, 5)}},
        },
    ]
    solve_problems(ten)


if __name__ == "__main__":
    main()
