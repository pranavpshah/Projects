from heapq import heappush, heappop  # Recommended.
import numpy as np
import itertools

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.
from .node import Node
# from occupancy_map import OccupancyMap
# from node import Node

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    nodes_expanded = 0

    occupancy_grid = occ_map.map

    x_range = occupancy_grid.shape[0]
    y_range = occupancy_grid.shape[1]
    z_range = occupancy_grid.shape[2]

    #creating an empty array with the shape same as occupancy map
    nodes = np.empty(shape = (x_range, y_range, z_range), dtype = 'object')

    #assigning a Node class object to each and every voxel in the map
    for i in range(x_range):
        for j in range(y_range):
            for k in range(z_range):
                if(astar == True):
                    nodes[i,j,k] = Node(heuristic=np.linalg.norm(np.array([i,j,k]) - np.array(goal_index)),index = np.array([i,j,k]))
                else:
                    nodes[i,j,k] = Node(heuristic=0,index = np.array([i,j,k]))

    #assigning different costs and heuristics to the start and goal voxels
    if(astar == True):
        nodes[start_index[0],start_index[1],start_index[2]] = Node(cost_to_come=0, heuristic= np.linalg.norm(np.array(start_index) - np.array(goal_index)), index = np.array(start_index))
        nodes[goal_index[0],goal_index[1],goal_index[2]] = Node(heuristic=np.linalg.norm(np.array(goal_index) - np.array(goal_index)),index = np.array(goal_index))
    else:
        nodes[start_index[0],start_index[1],start_index[2]] = Node(cost_to_come=0,heuristic=0,index = np.array(start_index))
        nodes[goal_index[0],goal_index[1],goal_index[2]] = Node(heuristic=0,index = np.array(goal_index))

    #initializing a list for the queue
    to_be_visited = []
    start_node = nodes[start_index[0],start_index[1],start_index[2]]
    start_node.parent = None
    #pushing the start_node to the heap queue
    heappush(to_be_visited, start_node)
    start_node.visited = True

    end_node = nodes[goal_index[0],goal_index[1],goal_index[2]]
    
    #initializing a actions list to define 26 neighbors
    actions = np.array([0,0,0])

    #list of actions to take from the center node to reach 26 neighbors
    for direction in itertools.product((-1, 0, 1), repeat=3):
        if(direction == np.array([0,0,0])).all():
            continue
        actions = np.vstack((actions, direction))

    
    #runnning a loop till the heap queue is empty
    while len(to_be_visited) > 0:

        #pop the node with lowest cost from the heap queue
        current_node = heappop(to_be_visited)

        nodes_expanded += 1

        pos = current_node.index

        #check if goal is reached
        if((pos == end_node.index).all()):
            break

        #loop through all the neighbors of a particular node
        for action in actions:
            i = action[0]
            j = action[1]
            k = action[2]

            #ignore if it is the current node
            if(i==0 and j==0 and k==0):
                continue

            new_index = (pos[0]+i, pos[1]+j, pos[2]+k)

            #check if a particular node index is valid given a current node
            if(occ_map.is_valid_index(new_index)):
                next_node = nodes[new_index[0], new_index[1], new_index[2]]
            else:
                continue

            #check if a given node is occupied by an obstacle
            if(occ_map.is_occupied_index(new_index)):
                continue

            #cost computation for neighboring node
            if(astar == True):
                computed_cost = np.linalg.norm(next_node.index - current_node.index) + current_node.cost_to_come + next_node.heuristic
            else:
                computed_cost = np.linalg.norm(next_node.index - current_node.index) + current_node.cost

            #checking if the computed cost of the neighboring node is less that previous cost
            if(computed_cost < next_node.cost):
                next_node.cost = computed_cost
                next_node.cost_to_come = np.linalg.norm(next_node.index - current_node.index) + current_node.cost_to_come
                next_node.parent = current_node
            
            #check if node is already in heap queue 
            if(next_node.visited == False):
                heappush(to_be_visited, next_node)
                next_node.visited = True

        
    #check if no path has been founded
    if(len(to_be_visited) == 0):
        path = None
        return path, nodes_expanded
    
    #retrace path from goal node to start node
    last_node = end_node
    parent_node = end_node.parent
    path = goal
    while(parent_node is not None):
        last_node = last_node.parent
        path = np.vstack((occ_map.index_to_metric_center(last_node.index), path))
        parent_node = last_node.parent
    
    path[0] = start
    
    return path, nodes_expanded

