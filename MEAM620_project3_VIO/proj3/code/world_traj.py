import numpy as np
import pdb

# from .graph_search import graph_search
# from proj3.code.graph_search import graph_search
from graph_search import graph_search

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.21, 0.21, 0.21])
        self.margin = 0.5

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1,3)) # shape=(n_pts,3)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

        # STUDENT CODE HERE
        self.waypoints = self.path
        #pdb.set_trace()
        
        #path pruning, considering every 6th point from the original path
        self.waypoints = np.concatenate((self.waypoints[0:-1:7], self.path[-1].reshape((1,3))), axis = 0)
        
        
        self.points = self.waypoints.copy()
        end = self.waypoints.shape[0]

        #defining start and end points for each segment
        self.start_points = self.waypoints[0:end-1, :].copy()
        self.end_points = self.waypoints[1:,:].copy()

        #computing vectors corresponding to start of each segment
        vectors_temp = self.waypoints[1:,:] - self.waypoints[0:end-1,:]
        self.vectors = vectors_temp/(np.linalg.norm(vectors_temp, axis=1).reshape((end-1,1)))
        self.distances = np.linalg.norm(vectors_temp, axis = 1) #length of each segment

        self.avg_speed = 2.9 #defining average speed for each segment in m/s
        self.duration = self.distances/self.avg_speed #duration of each segment computed considering average speed of 3.6 m/s

        count = 0
        for i in range(1,len(self.vectors) - 1):
            vec1 = self.vectors[i]
            vec2 = self.vectors[i+1]
            theta = np.arccos(np.dot(vec1,vec2))
            if(theta > 0.4):
                self.duration[i] = self.duration[i]*1.4*np.sin(theta)
                count += 1
            else:
                self.duration[i] = self.duration[i]*np.cos(theta)

        # pdb.set_trace()
       
        # size = self.duration.shape[0]
        self.duration[0] = self.duration[0]*2
        self.duration[-1] = self.duration[-1]*2


        self.lhs_mat = np.zeros((6*self.vectors.shape[0], 6*self.vectors.shape[0])) #initializing lhs matrix for Ax = B, i.e., A matrix
        self.rhs_mat = np.zeros((6*self.vectors.shape[0], 3))   #initializing rhs matrix for Ax = B, i.e., B matrix

        #initializing variables to keep track of rows columns and waypoints
        j=0
        i=0
        k = 0
        flag = True

        #position contraints, 2 per segment
        while flag:
            self.rhs_mat[i] = self.start_points[j]
            self.rhs_mat[i+1] = self.end_points[j]

            self.lhs_mat[i, k:k+6] = np.array([0,0,0,0,0,1])
            self.lhs_mat[i+1, k:k+6] = np.array([self.duration[j]**5, self.duration[j]**4, self.duration[j]**3, self.duration[j]**2, self.duration[j], 1])

            i = i+2
            j = j+1
            k = k+6

            #end loop when all waypoints covered
            if(j == self.vectors.shape[0]):
                flag = False
            
            
        j = j-1

        #velocity at start of path and end of path 
        k = 0
        self.lhs_mat[i, k:k+6] = np.array([0,0,0,0,1,0])    #for velocity at start node
        self.rhs_mat[i] = np.array([0,0,0])
        k = self.lhs_mat.shape[1]-6     #for velocity at end node
        self.lhs_mat[i+1, k:k+6] = np.array([5*(self.duration[j]**4), 4*(self.duration[j]**3), 3*(self.duration[j]**2), 2*self.duration[j], 1, 0])
        self.rhs_mat[i+1] = np.array([0,0,0])

        i = i+2
        #acceleration at start of path and end of path 
        k=0
        self.lhs_mat[i, k:k+6] = np.array([0,0,0,2,0,0])    #for acceleration at start node
        self.rhs_mat[i] = np.array([0,0,0])
        k = self.lhs_mat.shape[1]-6     #for acceleration at end node
        self.lhs_mat[i+1, k:k+6] = np.array([20*(self.duration[j]**3), 12*(self.duration[j]**2), 6*(self.duration[j]), 2, 0, 0])
        self.rhs_mat[i+1] = np.array([0,0,0])

        i = i+2
        j = 0
        k = 0

        flag = True

        #continuity constraints on the derivatives of the 5th order polynomial of minimum jerk trajectory
        while flag:
            
            #velocity continuity
            self.lhs_mat[i, k:k+12] = np.array([5*(self.duration[j]**4), 4*(self.duration[j]**3), 3*(self.duration[j]**2), 2*self.duration[j], 1, 0, 0,0,0,0,-1,0])
            self.rhs_mat[i] = np.array([0,0,0])

            #acceleration continuity
            self.lhs_mat[i+1, k:k+12] = np.array([20*(self.duration[j]**3), 12*(self.duration[j]**2), 6*(self.duration[j]), 2, 0, 0, 0,0,0,-2,0,0])
            self.rhs_mat[i+1] = np.array([0,0,0])

            #jerk continuity
            self.lhs_mat[i+2, k:k+12] = np.array([60*(self.duration[j]**2), 24*(self.duration[j]), 6, 0, 0, 0, 0,0,-6,0,0,0])
            self.rhs_mat[i+2] = np.array([0,0,0])

            #snap continuitiy
            self.lhs_mat[i+3, k:k+12] = np.array([120*(self.duration[j]), 24, 0, 0, 0, 0, 0,-24,0,0,0,0])
            self.rhs_mat[i+3] = np.array([0,0,0])

            i = i+4
            j = j+1
            k = k+6

            #end loop when all intermediate waypoints are covered
            if(j == self.vectors.shape[0]-1):
                flag = False

        #computing coefficient matrix using : x = inv(A) x B
        self.coeff_mat = np.linalg.inv(self.lhs_mat) @ self.rhs_mat

        #segment tracking, flight time and coefficient matrix pointer variables
        self.prev_time = 0
        self.pointer = 0
        self.segment_start_time = 0
        self.segment_counter = self.vectors.shape[0]
        self.coeff_mat_counter = 0

        self.prev_x = np.zeros((3,1))
        self.flag = True

    def update(self, t):
            """
            Given the present time, return the desired flat output and derivatives.

            Inputs
                t, time, s
            Outputs
                flat_output, a dict describing the present desired flat outputs with keys
                    x,        position, m
                    x_dot,    velocity, m/s
                    x_ddot,   acceleration, m/s**2
                    x_dddot,  jerk, m/s**3
                    x_ddddot, snap, m/s**4
                    yaw,      yaw angle, rad
                    yaw_dot,  yaw rate, rad/s
            """
            x        = np.zeros((3,))
            x_dot    = np.zeros((3,))
            x_ddot   = np.zeros((3,))
            x_dddot  = np.zeros((3,))
            x_ddddot = np.zeros((3,))
            yaw = 0
            yaw_dot = 0

            # STUDENT CODE HERE

            #checking if quad is in the current segment time duration
            if(t < np.sum(self.duration[:self.pointer+1])):
                del_t = t - self.segment_start_time# computing delta_t, to be used to calculate values using the 5th order polynomial

                #Computing desired position
                x = np.array([[del_t**5, del_t**4, del_t**3, del_t**2, del_t, 1]]) @ self.coeff_mat[self.coeff_mat_counter:self.coeff_mat_counter+6]
                x = x.reshape((3,))
                
                #computing desired velocity
                x_dot = np.array([[5*(del_t**4), 4*(del_t**3), 3*(del_t**2), 2*del_t, 1, 0]]) @ self.coeff_mat[self.coeff_mat_counter:self.coeff_mat_counter+6]
                x_dot = x_dot.reshape((3,))

                #computing desired acceleration
                x_ddot = np.array([[20*(del_t**3), 12*(del_t**2), 6*del_t, 2, 0, 0]]) @ self.coeff_mat[self.coeff_mat_counter:self.coeff_mat_counter+6]
                x_ddot = x_ddot.reshape((3,))

                #computing desired jerk
                x_dddot = np.array([[60*(del_t**2), 24*del_t, 6, 0, 0, 0]]) @ self.coeff_mat[self.coeff_mat_counter:self.coeff_mat_counter+6]
                x_dddot = x_dddot.reshape((3,))

                #computing desired snap
                x_ddddot = np.array([[120*del_t, 24, 0, 0, 0, 0]]) @ self.coeff_mat[self.coeff_mat_counter:self.coeff_mat_counter+6]
                x_ddddot = x_ddddot.reshape((3,))

                self.prev_x = x
                self.prev_time = t
            
            #checking if the time for a given segment has passed, then start new segment
            elif((t>=np.sum(self.duration[:self.pointer+1]) and t<np.sum(self.duration[:self.pointer+2]))):
                self.pointer += 1   #point to next segment
                self.segment_counter -= 1
                self.segment_start_time = self.prev_time    #change segment start time
                self.coeff_mat_counter = self.coeff_mat_counter + 6     #used coefficients corresponding to the new segment
                if(self.pointer == self.end_points.shape[0]):
                    #make bot hover at end node after all segments and time durations completed
                    x = self.waypoints[-1,:].copy()
                    x_dot = np.zeros((3,))
                    x_ddot = np.zeros((3,))
                    self.flag = False
                else:
                    #assign values for new segment start

                    #desired position, start of segment
                    x = self.waypoints[self.pointer,:].copy()

                    #delta_t
                    del_t = t - np.sum(self.duration[:self.pointer+1])

                    #desired velocity at start of new segment
                    x_dot = np.array([[5*(del_t**4), 4*(del_t**3), 3*(del_t**2), 2*del_t, 1, 0]]) @ self.coeff_mat[self.coeff_mat_counter:self.coeff_mat_counter+6]
                    x_dot = x_dot.reshape((3,))

                    #desired acceleration at start of new segment
                    x_ddot = np.array([[20*(del_t**3), 12*(del_t**2), 6*del_t, 2, 0, 0]]) @ self.coeff_mat[self.coeff_mat_counter:self.coeff_mat_counter+6]
                    x_ddot = x_ddot.reshape((3,))

                    #desired jerk at start of new segment
                    x_dddot = np.array([[60*(del_t**2), 24*del_t, 6, 0, 0, 0]]) @ self.coeff_mat[self.coeff_mat_counter:self.coeff_mat_counter+6]
                    x_dddot = x_dddot.reshape((3,))

                    #desired snap at the start of new segment
                    x_ddddot = np.array([[120*del_t, 24, 0, 0, 0, 0]]) @ self.coeff_mat[self.coeff_mat_counter:self.coeff_mat_counter+6]
                    x_ddddot = x_ddddot.reshape((3,))

            else:
                #all segments completed, hover at end node (goal position)
                x = self.waypoints[-1,:].copy()
            

            flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                            'yaw':yaw, 'yaw_dot':yaw_dot}
            return flat_output