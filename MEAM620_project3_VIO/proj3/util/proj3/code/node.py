import numpy as np

#class Node to save paramters of each node

class Node:
    
    def __init__(self, cost_to_come=np.inf, heuristic=0, index=np.array([0,0,0]), parent=None):
        self.cost_to_come = cost_to_come   
        self.heuristic = heuristic
        self.index = index
        self.parent = parent
        self.visited = False
        self.cost = cost_to_come + heuristic

    def __lt__(self, other):
        return self.cost < other.cost
    
    def __gt__(self, other):
        return self.cost > other.cost

    def __eq__(self, other):
        return self.cost == other.cost

    

    