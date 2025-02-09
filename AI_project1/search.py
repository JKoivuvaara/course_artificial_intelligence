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


def getPath(problem, actions_taken: []) -> []:
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
    used by graphSearch and weightedGraphSearch
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
    reached = dict()  # TODO: change this to a set
    actions_taken = []
    frontier.push( (problem.getStartState(), None, 0, None) )  # (node, direction, cost, parent_node)

    while not frontier.isEmpty():
        action = frontier.pop()
        node, direction, path_cost, parent = action

        if reached.get(node):
            continue  # if node has been visited
        reached[node] = node

        if direction:  # skip first iteration
            actions_taken.append(action)

        if problem.isGoalState(node):
            return getPath(problem, actions_taken)

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


# def getPath_cornersProblem(problem, actions_taken: []):
#     """
#     for solving cornersProblem, because the other bfs solution constructs the path in a way that is not
#     compatible with cornerProblem.
#     """
#     path = []
#     i = len(actions_taken) - 1
#     while i >= 0:  # loop backwards over the list, beginning from the last action
#         action = actions_taken[i]  # (this_node, how_we_got_here, cost, parent)
#         direction = action[1]
#         parent = action[3]
#         path.append(direction)
#
#         # check previous nodes until we find the parent's action
#         i = i - 1
#         while i >= 0 and parent != actions_taken[i][0]:
#             i = i - 1
#
#     path.reverse()
#     return path


# def expand_cornersProblem(problem, node, path_cost, explored_nodes):
#     """
#     for solving cornersProblem, because the other bfs solution constructs the path in a way that is not
#     compatible with cornerProblem.
#     returns: Generator[child_node, direction_to_child_node]
#     """
#     for successor in problem.getSuccessors(node):
#         # (next_node, direction, cost, parent_node)
#         if successor[0] not in explored_nodes:
#
#             yield successor
#
#     for successor in problem.getSuccessors(node):
#         next_node, direction, action_cost = successor
#         if next_node not in explored_nodes:
#             yield next_node, direction, path_cost + action_cost, node


def graphSearch_cornersProblem(problem, frontier, start_location, goal_locations):
    """
    for solving cornersProblem, actually almost identical with graphSearch...
    TODO: replace original graphSearch-function with this function

    This function allows for multiple starting locations for the search.
    @param goal_locations: list of locations being searched
    @param start_location: start search here
    @param problem: e.g. cornersProblem
    @param frontier: queue for bfs, stack for dfs
    """
    explored_nodes = set()
    # set of explored nodes {(a,b), (c,d)... }
    # value: (neighbour, direction_to_neighbour)
    actions_taken = []

    # return this
    connections_found = []  # [(start, goal, path_to_goal), ...]

    actions_taken = []  # [(this_node, how_we_got_here, cost, parent), ...]

    # push starting locations to queue
    frontier.push( (start_location, None, 0, None) )  # (this_node, how_we_got_here, path_cost, parent)

    """ Begin main loop """
    while not frontier.isEmpty():
        """explore next node in queue"""
        node, direction, path_cost, parent = frontier.pop()  # (this_node, how_we_got_here, cost, parent)

        """if the node has already been explored, continue from the beginning"""
        if node in explored_nodes:
            continue

        """if not, add the current node to the set of explored nodes"""
        explored_nodes = explored_nodes | {node}
        # print(f"set = {explored_nodes}")

        """keep track of actions taken"""
        if direction:  # skip first round
            actions_taken.append((node, direction, path_cost, parent))

        """when a goal_location is found, append the connection"""
        if node in goal_locations:
            path_to_goal = getPath(problem, actions_taken)
            connections_found.append((start_location, node, path_cost, path_to_goal))

        """check if all goal_locations have been found"""
        if len(goal_locations) == len(connections_found):
            return connections_found

        """expand from current node"""
        for child in expand(problem, node, path_cost, explored_nodes):
            frontier.push(child)

    print("A goal is not reachable, might be a bug...")
    util.raiseNotDefined()


"""Function for recursively finding all tours in a graph. Generated by Copilot, slightly modified by me."""
def find_all_tours(nodes):
    """
    Basically travelling salesman through bruteforce. However, corners problem only has 5 nodes.
    The amount of possible paths in a complete graph is (n-1)! = (5-1)! = 4*3*2*1 = 24.

    Currently, assumes that the graph is complete, aka all nodes are connected to all other nodes
    @param nodes: node[0] is the starting node for the tours
    """
    start_node = nodes[0]
    all_tours = []

    # recursive search method
    def generate_tours(current_node, visited: set, path):
        if len(visited) == len(nodes):  # recursion end condition
            if start_node in nodes:
                all_tours.append(path)
            return

        for neighbor in nodes:
            if neighbor not in visited:
                generate_tours(neighbor, visited | {neighbor}, path + [neighbor])

    # start recursive search
    generate_tours(start_node, {start_node}, [start_node])

    print(f"found {len(all_tours)} tours")
    for tour in all_tours:
        print(tour)

    return all_tours

# # for debugging find_all_tours
# nodes = ['A', 'B', 'C', 'D']
# connections = {
#     'A': ['B', 'C', 'D'],
#     'B': ['A', 'C', 'D'],
#     'C': ['A', 'B', 'D'],
#     'D': ['A', 'B', 'C']
# }
# tours = find_all_tours(nodes)

def calculateTourCost(tour: list, connections: list) -> int:
    # if list of connections becomes very large, replace it with a dict for faster search
    cost = 0
    for i in range(len(tour) - 1):
        start_node, end_node = tour[i], tour[i+1]
        required_nodes = {start_node, end_node}
        for c in connections:
            nodes_in_this_connection = {c[0], c[1]}  # a set with start node and end node
            if required_nodes == nodes_in_this_connection:
                cost += c[2]
                break  # connection found
    return cost

def reversePath(path):
    # print(f"reversing path {path}")
    reverse = {
        'North': 'South',
        'South': 'North',
        'East': 'West',
        'West': 'East'
    }
    reversed_path = []
    for step in path:
        reversed_path.append(reverse.get(step))
    reversed_path.reverse()
    # print(f"reversed path is {reversed_path}")
    return reversed_path

def combinePaths(tour, connections):
    """
    Some paths need to be reversed.
    I'm assuming the only directions in use are 'North', 'East', 'South' and 'West'.
    """
    combined_path = []
    for i in range(len(tour) - 1):
        start_node, end_node = tour[i], tour[i+1]
        required_nodes = {start_node, end_node}
        for c in connections:
            nodes_in_this_connection = {c[0], c[1]}  # a set with start node and end node
            if required_nodes == nodes_in_this_connection:
                path = c[3]
                if start_node == c[1]:  # path needs to be reversed
                    path = reversePath(path)
                # print(f"adding path {path}")
                for step in path:
                    combined_path.append(step)

    return combined_path


def findShortestPath(start_connections, connections):
    # connection: (start, end, path_cost, path_to_goal)
    # connections: [connection1, connection2...]
    nodes = []  # list of nodes [(a,b), (c,d)...]
    tours = []  # [[node1 ... node_n], [node1 ... node_n-1]...] tour is a path that goes through all nodes once
    # TODO: if the amount of nodes is large, consider using a dictionary for faster retrieval of elements

    """add all nodes to the nodes list"""
    nodes.append(start_connections[0][0])
    for i in range(len(start_connections)):
        nodes.append(start_connections[i][1])
    print(f"nodes in the graph: {nodes}")

    """find all tours by brute force"""
    tours = find_all_tours(nodes)

    """find the shortest tour"""
    lowest_tour_cost = 9999999999
    shortest_tour = None
    for tour in tours:
        cost = calculateTourCost(tour, connections)
        if cost < lowest_tour_cost:
            lowest_tour_cost = cost
            shortest_tour = tour
    print(f"Shortest tour is {lowest_tour_cost} steps long: {shortest_tour}")

    """now we can reconstruct the complete path"""
    path = combinePaths(shortest_tour, connections)
    print(f"The shortest path is: {path}")
    return path

    util.raiseNotDefined()
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    ### corners problem ###
    # uses a slightly modified graphSearch function
    if hasattr(problem, "iam") and problem.iam == "CornersProblem":
        all_connections = []  # [(start, end, path_cost, path_to_goal),...]
        start_connections = []

        """start bfs from every start location separately"""
        all_locations = problem.getStartState()
        goal_locations = list.copy(all_locations)
        for i, start_location in enumerate(all_locations):
            goal_locations.remove(start_location)  # if A->B is found, dont look for B->A
            new_connections = graphSearch_cornersProblem(problem, util.Queue(), start_location, goal_locations)
            if i == 0:
                start_connections = new_connections
            for nc in new_connections:
                all_connections.append(nc)
            # print("All connections found ", new_connections)
            # print(f"connections in total: {len(new_connections)}")

        # print(start_connections)
        # print(all_connections)

        return findShortestPath(start_connections, all_connections)
    ### corners problem section ends ###


    frontier = util.Queue()  # First in first out queue
    path = graphSearch(problem, frontier)
    return path


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def weightedGraphSearch(problem, frontier, heuristic=nullHeuristic) -> []:
    """
    used by ucs, astar
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
            return getPath(problem, actions_taken)

        for child in expand(problem, node, path_cost, reached):
            frontier.update(child, child[2] + heuristic(child[0], problem))

    return []  # no path to goal found


def uniformCostSearch(problem):
    """Search the node of the smallest total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()  # First in first out queue
    path = weightedGraphSearch(problem, frontier)
    return path


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()  # First in first out queue
    path = weightedGraphSearch(problem, frontier, heuristic)
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
