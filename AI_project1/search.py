# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def get_path(problem, actions_taken: []) -> []:
    path = []
    i = len(actions_taken) - 1
    while i >= 0:  # loop backwards over the list, beginning from the last action
        action = actions_taken[i]  # (node, direction, cost, parent)
        direction = action[1]
        parent = action[3]
        path.append(direction)

        # check previous nodes until we find the parent's action
        i = i - 1
        while i >= 0 and parent != actions_taken[i][0]:
            i = i - 1

    path.reverse()
    return path


def expand(problem, node, path_cost, reached):
    """
    returns: Generator[child_node, direction, cost, parent_node]
    """
    for successor in problem.getSuccessors(node):
        next_node, direction, action_cost = successor
        if next_node not in reached:
            yield next_node, direction, path_cost + action_cost, node


def graphSearch(problem, frontier) -> []:
    """
    used by dfs, bfs
    @param problem:
    @param frontier: queue, needs to have functions push(), pop()
    """
    reached = dict()
    actions_taken = []
    frontier.push( (problem.getStartState(), None, 0, None) )  # (node, direction, cost, parent_node)

    while not frontier.isEmpty():
        action = frontier.pop()
        node, direction, path_cost, parent = action

        if reached.get(node):
            continue
        reached[node] = node

        if direction:  # skip first iteration
            actions_taken.append(action)

        if problem.isGoalState(node):
            return get_path(problem, actions_taken)

        for child in expand(problem, node, path_cost, reached):
            frontier.push(child)

    return []  # no path to goal found


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    frontier = util.Stack()  # Last in first out queue
    path = graphSearch(problem, frontier)
    return path


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    frontier = util.Queue()  # First in first out queue
    path = graphSearch(problem, frontier)
    return path


def weightedGraphSearch(problem, frontier) -> []:
    """
    used by ucs
    @param problem:
    @param frontier: queue, needs to have functions push(), pop(), update()
    """
    reached = dict()
    actions_taken = []
    frontier.push((problem.getStartState(), None, 0, None), 0)  # (node, direction, cost, parent_node), priority

    while not frontier.isEmpty():
        action = frontier.pop()
        node, direction, path_cost, parent = action

        if reached.get(node):
            continue
        reached[node] = node

        if direction:  # skip first iteration
            actions_taken.append(action)

        if problem.isGoalState(node):
            return get_path(problem, actions_taken)

        for child in expand(problem, node, path_cost, reached):
            frontier.update(child, child[2])

    return []  # no path to goal found


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()  # First in first out queue
    path = weightedGraphSearch(problem, frontier)
    return path


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
