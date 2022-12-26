import numpy as np
from scipy.signal import cont2discrete
from typing import List, Tuple
import quad_sim
import pdb


class iLQR(object):

    # Observations:
    # 1. Do we want to trajectory optimize to a goal (it could be the next one, or the last one)   
    #    or do we want to pass a series of goal states that we track for different units of discrete time
    #       --> Do we solve it piecewise or what


    def __init__(self, x_goal: np.ndarray, N: int, dt: float, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray):
        """
        Constructor for the iLQR solver
        :param N: iLQR horizon
        :param dt: timestep
        :param Q: weights for running cost on state
        :param R: weights for running cost on input
        :param Qf: weights for terminal cost on input
        """

        # Quadrotor dynamics parameters
        self.m = 1
        self.a = 0.25
        self.I = 0.0625
        self.nx = 6
        self.nu = 2

        # iLQR constants
        self.N = N
        self.dt = dt

        # Solver parameters
        self.alpha = 1e-2         
        self.max_iter = 1e3
        self.tol = 1e-4

        # target state
        self.x_goal = x_goal
        self.u_goal = 0.5 * 9.81 * np.ones((2,))

        # Cost terms
        self.Q = Q
        self.R = R
        self.Qf = Qf

    def total_cost(self, xx, uu):
        J = sum([self.running_cost(xx[k], uu[k]) for k in range(self.N - 1)])
        return J + self.terminal_cost(xx[-1])


    # Not used for us?
    def get_linearized_dynamics(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        
        """
        :param x: quadrotor state
        :param u: input
        :return: A and B, the linearized continuous quadrotor dynamics about some state x
        """
        m = self.m
        a = self.a
        I = self.I
        A = np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, -np.cos(x[2]) * (u[0] + u[1]) / m, 0, 0, 0],
                      [0, 0, -np.sin(x[2]) * (u[0] + u[1]) / m, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [-np.sin(x[2]) / m, -np.sin(x[2]) / m],
                      [np.cos(x[2]) / m, np.cos(x[2]) / m],
                      [a / I, -a / I]])

        return A, B


        



    # Not underanding this
    def get_linearized_discrete_dynamics(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x: state
        :param u: input
        :return: the discrete linearized dynamics matrices, A, B as a tuple
        """
        # A, B = self.get_linearized_dynamics(x, u)
        # C = np.eye(A.shape[0])
        # D = np.zeros((A.shape[0],))
        # [Ad, Bd, _, _, _] = cont2discrete((A, B, C, D), self.dt)
        # return Ad, Bd
        # DAFAQ IS GOING ON HERE?
        dt = self.dt
        v_r = x[2]
        theta_r = u[0]
        A_d = np.array([[1, 0, -v_r*np.sin(theta_r)*dt],
						[0, 1, v_r*np.cos(theta_r)*dt],
						[0, 0, 1]])

        B_d = np.array([[np.cos(theta_r)*dt, 0],
						[np.sin(theta_r)*dt, 0],
						[0, dt]])
        
        return A_d, B_d



    def running_cost(self, xk: np.ndarray, uk: np.ndarray) -> float:
        #goal for each state?
        
        """
        :param xk: state
        :param uk: input
        :return: l(xk, uk), the running cost incurred by xk, uk
        """

        # Standard LQR cost on the goal state
        lqr_cost = 0.5 * ((xk - self.x_goal).T @ self.Q @ (xk - self.x_goal) +
                          (uk - self.u_goal).T @ self.R @ (uk - self.u_goal))

        return lqr_cost

    def grad_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: [ᵀ∂l/∂x, ∂l/∂uᵀ]ᵀ, evaluated at xk, uk
        """
        grad = np.zeros((8,))

        #TODO: Compute the gradient
        Qdx = self.Q@(xk - self.x_goal);
        Rdx = self.R@(uk - self.u_goal);
        grad[0:Qdx.shape[0]] = Qdx.reshape((Qdx.shape[0],));
        grad[Qdx.shape[0]:] = Rdx.reshape((Rdx.shape[0],));

        return grad;

    def hess_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: The hessian of the running cost
        [[∂²l/∂x², ∂²l/∂x∂u],
         [∂²l/∂u∂x, ∂²l/∂u²]], evaluated at xk, uk
        """


        H = np.zeros((self.nx + self.nu, self.nx + self.nu))
        H[:self.Q.shape[0], :self.Q.shape[0]] = self.Q;
        H[self.Q.shape[0]:, self.Q.shape[0]:] = self.R;

        # TODO: Compute the hessian


        return H

    def terminal_cost(self, xf: np.ndarray) -> float:
        """
        :param xf: state
        :return: Lf(xf), the running cost incurred by xf
        """
        return 0.5*(xf - self.x_goal).T @ self.Qf @ (xf - self.x_goal)

    def grad_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        """
        :param xf: final state
        :return: ∂Lf/∂xf
        """
        
        grad = np.zeros((self.nx));
        grad = (self.Qf @ (xf - self.x_goal)).reshape((self.nx,));

        # TODO: Compute the gradient

        return grad
        
    def hess_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        """
        :param xf: final state
        :return: ∂²Lf/∂xf²
        """ 

        H = np.zeros((self.nx, self.nx))

        # TODO: Compute H
        H = self.Qf;
        return H

    def forward_pass(self, xx: List[np.ndarray], uu: List[np.ndarray], dd: List[np.ndarray], KK: List[np.ndarray]) -> \
            Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        :param xx: list of states, should be length N
        :param uu: list of inputs, should be length N-1
        :param dd: list of "feed-forward" components of iLQR update, should be length N-1
        :param KK: list of "Feedback" LQR gain components of iLQR update, should be length N-1
        :return: A tuple (xtraj, utraj) containing the updated state and input
                 trajectories after applying the iLQR forward pass
        """

        xtraj = [np.zeros((self.nx,))] * self.N
        utraj = [np.zeros((self.nu,))] * (self.N - 1)
        xtraj[0] = xx[0]

        for i in range(0,len(xx)-1):
            dx = xtraj[i] - xx[i];
            u = uu[i] + KK[i]@dx + self.alpha*dd[i];
            xtraj[i+1] = quad_sim.F(xtraj[i],u,self.dt);
            utraj[i] = u;

        # TODO: compute forward pass

        print("In forward pass")

        return xtraj, utraj

    def backward_pass(self,  xx: List[np.ndarray], uu: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        :param xx: state trajectory guess, should be length N
        :param uu: input trajectory guess, should be length N-1
        :return: KK and dd, the feedback and feedforward components of the iLQR update
        """
        dd = [np.zeros((self.nu,))] * (self.N - 1)
        KK = [np.zeros((self.nu, self.nx))] * (self.N - 1)

        
        # TODO: compute backward pass


        for i in range(len(xx)-2, -1, -1):
            #A,B = self.get_linearized_discrete_dynamics(xx[i] - self.x_goal, uu[i] - self.u_goal);
            A,B = self.get_linearized_discrete_dynamics(xx[i], uu[i]);
            if(i == len(xx)-2):
                gk1 = self.grad_terminal_cost(xx[i+1]);
                hk1 = self.hess_terminal_cost(xx[i+1]);
            else:
                gk1 = Qx - KK[i+1].T @ Quu @ dd[i+1];
                hk1 = Qxx - KK[i+1].T@Quu@KK[i+1];

            g = self.grad_running_cost(xx[i],uu[i]);
            h = self.hess_running_cost(xx[i],uu[i]);
            
            #pdb.set_trace();
            Qu = g[-2:] + B.T@gk1;
            Quu = h[-2:,-2:] + B.T@hk1@B;
            Qux = B.T@hk1@A;
            Qxx = h[:6,:6] + A.T@hk1@A;
            Qx = g[:6] + A.T@gk1;



            dd[i] = -np.linalg.inv(Quu)@Qu;
            KK[i] = -np.linalg.inv(Quu)@Qux;
            #pdb.set_trace();

        return dd, KK

    def calculate_optimal_trajectory(self, x: np.ndarray, uu_guess: List[np.ndarray]) -> \
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

        """
        Calculate the optimal trajectory using iLQR from a given initial condition x,
        with an initial input sequence guess uu
        :param x: initial state
        :param uu_guess: initial guess at input trajectory
        :return: xx, uu, KK, the input and state trajectory and associated sequence of LQR gains
        """
        assert (len(uu_guess) == self.N - 1)

        # Get an initial, dynamically consistent guess for xx by simulating the quadrotor
        xx = [x]
        for k in range(self.N-1):
            xx.append(quad_sim.F(xx[k], uu_guess[k], self.dt))

        Jprev = np.inf
        Jnext = self.total_cost(xx, uu_guess)
        uu = uu_guess
        KK = None

        i = 0
        print(f'cost: {Jnext}')
        while np.abs(Jprev - Jnext) > self.tol and i < self.max_iter:
            dd, KK = self.backward_pass(xx, uu)
            xx, uu = self.forward_pass(xx, uu, dd, KK)

            Jprev = Jnext
            Jnext = self.total_cost(xx, uu)
            print(f'cost: {Jnext}')
            i += 1
        print(f'Converged to cost {Jnext}')
        return xx, uu, KK
