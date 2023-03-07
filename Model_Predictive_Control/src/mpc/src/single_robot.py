#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos
import math
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.linalg import solve_continuous_are

from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.osqp import OsqpSolver
from pydrake.solvers.snopt import SnoptSolver
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
import pydrake.symbolic as sym

from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables

x_other_robot = [];

class Robot(object):
	def __init__(self, Q, R, Qf, bot_radius, epsilon, robot_id):
		self.g = 9.81
		self.m = 1
		self.a = 0.25
		self.I = 0.0625
		self.Q = Q # Shape based on state
		self.R = R
		self.Qf = Qf

		# Input limits
		# self.umin = np.array([-0.26, -2.84])
		# self.umax = np.array([0.26, 2.84])
		
		self.umin = np.array([-0.26, -2.2])
		self.umax = np.array([0.26, 2.2])


		self.nx = 3 # x, y, theta
		self.nu = 2 # v, theta dot(omega)
		self.bot_radius = bot_radius
		self.epsilon = epsilon
		self.robot_id = robot_id - 1	# making it start from 0 index

		# iLQR data

	def continuous_time_full_dynamics(self, x, u):
		# Dynamics for the quadrotor
		# TODO: Pranav add dynamics

		theta = -x[2]
		v = -u[0]
		w = -u[1]

		xdot = np.array([v*np.cos(theta),
						v*np.sin(theta),
						w])
		return xdot

	"""
	Keep commented for now, will remove later
		def continuous_time_linearized_dynamics(self, xr, ur):
			# TODO: Pranav add dynamics
			# Dynamics linearized at the fixed point
			# This function returns A and B matrix

			# theta_r (reference theta)
			# v _r (reference velocity)
			# T (sampling period)

			theta_r = xr[2]
			v_r = ur[0]

			A = np.array([[1, 0, -v_r*np.sin(theta_r)],
						[0, 1, v_r*np.cos(theta_r)],
						[0, 0, 1]])

			B = np.array([[np.cos(theta_r), 0],
						[np.sin(theta_r), 0],
						[0, 1]])

			return A, B

	"""

	def discrete_time_linearized_dynamics(self, xr, ur, dt):
		# CHECK PAPER

		# xr - Intermediate goal 
		# ur - Previous input

		# Discrete time version of the linearized dynamics at the fixed point
		# This function returns A and B matrix of the discrete time dynamics

		# A_c, B_c = self.continuous_time_linearized_dynamics()
		# A_d = np.identity(3) + A_c * T;
		# B_d = B_c * T;

		v_r = ur[0]
		theta_r = xr[2]

		A_d = np.array([[1, 0, -v_r*np.sin(theta_r)*dt],
						[0, 1, v_r*np.cos(theta_r)*dt],
						[0, 0, 1]])

		B_d = np.array([[np.cos(theta_r)*dt, 0],
						[np.sin(theta_r)*dt, 0],
						[0, dt]])

		return A_d, B_d


	###
		# TODO: MPC 
	###
	def add_initial_state_constraint(self, prog, x, x_current):
		prog.AddBoundingBoxConstraint(x_current, x_current, x[0])

	def add_input_saturation_constraint(self, prog, x, u, N):
		# Constraint on min and max input

		for ui in u:
			prog.AddBoundingBoxConstraint(self.umin[0], self.umax[0], ui[0])
			prog.AddBoundingBoxConstraint(self.umin[1], self.umax[1], ui[1])

	def add_dynamics_constraint(self, prog, x, u, N, xr, ur, x_current, dt):
		# Linearized turtlebot dynamcis
		# A, B = self.discrete_time_linearized_dynamics(xr, ur, dt)
		for i in range(N-1):
			# prog.AddLinearEqualityConstraint(A @ x[i,:] + B @ u[i,:] - x[i+1,:], np.zeros(self.nx))
			f = self.continuous_time_full_dynamics(x[i,:], u[i,:])
			prog.AddConstraint(x[i,0] + f[0]*dt - x[i+1,0], 0, 0)
			prog.AddConstraint(x[i,1] + f[1]*dt - x[i+1,1], 0, 0)
			prog.AddConstraint(x[i,2] + f[2]*dt - x[i+1,2], 0, 0)

	def add_cost(self, prog, xe, u, N):
		for i in range(N-1):
			prog.AddQuadraticCost(xe[i,:] @ self.Q @ xe[i,:].T)
			prog.AddQuadraticCost(u[i,:] @ self.R @ u[i,:].T)
		prog.AddQuadraticCost(xe[i,:] @ self.Qf @ xe[i,:].T)
	
	def collisionCostEvaluator(self, x, x_other):
		dist = np.sum((x[:,0:2] - x_other[0:2].reshape((1,-1)))**2, axis = 1).reshape((-1,1))
		check_dist = 1
		cost = dist[0,0]*0

		dist2 = dist**3
		cost = 1/(100*dist2)
		cost = np.sum(cost)

		return cost

	def add_collision_cost(self, prog, x, other_x):

		n_r = x.shape[0]
		n_c = x.shape[1]
		num_bots = len(other_x)

		for i in range(num_bots):

			def collisionCostHelper(vars):
				x = vars[:(n_r*n_c)].reshape((n_r, n_c))
				global x_other_robot;
				# x_other = vars[-3:,:]
				return self.collisionCostEvaluator(x, x_other_robot)

			if(i !=  self.robot_id):

				vars = np.zeros((n_r*n_c,), dtype = 'object')

				vars[:(n_r*n_c)] = x.reshape((-1,))
				temp = other_x[i].astype('object')
				# vars[-3:,:] = other_x[i].reshape((-1,1))
				#vars[-3:] = temp
				global x_other_robot;
				x_other_robot = temp

				prog.AddCost(collisionCostHelper, vars)

	# def add_collision_constraint(self, prog, x, x_current, other_x_current, dt):
	# 	D = (2*self.bot_radius) + self.epsilon
	# 	# dist = np.ones((x.shape[0], 1))*D
	# 	# zero_arr = np.zeros((x.shape[0],1))
	# 	for i in range(len(other_x_current)):
	# 		if(i != self.robot_id):
	# 			rob_dist = np.linalg.norm(x_current[:2] - other_x_current[i][:2])
	# 			print("Other bot pos : {} and rob_dist: {}".format(other_x_current[i], rob_dist))
	# 			other_pos = other_x_current[i][:2].reshape((1,-1))
	# 			# if(x_current[0] <= other_x_current[i][0]):
	# 			# 	temp1 = np.sum((other_pos - x[:,0:2])**2, axis = 1).reshape((-1,1))
	# 			# else:
	# 			# 	temp1 = np.sum((x[:,0:2] - other_pos)**2, axis = 1).reshape((-1,1))

	# 			temp1 = np.sum((other_pos - x[:,0:2])**2, axis = 1).reshape((-1,1))

	# 			# temp1 = np.linalg.norm((x[:,0:2] - other_pos), axis = 1).reshape((-1,1))
	# 			d_enforce = 1
	# 			prog.AddCost((1/(700* (temp1[0,:])))[0] );
	# 			if(rob_dist < d_enforce):
	# 				print("Hi there")
	# 				#prog.AddLinearCost((50*d_enforce - rob_dist)**2)

	# 				for j in range(temp1.shape[0]):
	# 					prog.AddConstraint(temp1[j,0] - D**2 >= 0)
	# 					# prog.AddCost(alpha * (p**(-temp1[j,0])))
	# 					# print(D, " ", temp1[j,0])
	# 				return 0.5
	# 			return dt


	def compute_mpc_feedback(self, x_current, x_r, u_r, other_x_current, T):
		'''
		This function computes the MPC controller input u
		'''

		# Parameters for the QP
		N = 20

		# Initialize mathematical program and declare decision variables
		prog = MathematicalProgram()
		x = np.zeros((N, self.nx), dtype="object")
		for i in range(N):
			x[i] = prog.NewContinuousVariables(self.nx, "x_" + str(i))
		u = np.zeros((N-1, self.nu), dtype="object")
		for i in range(N-1):
			u[i] = prog.NewContinuousVariables(self.nu, "u_" + str(i))

		# Add constraints and cost
		self.add_initial_state_constraint(prog, x, x_current)
		self.add_input_saturation_constraint(prog, x, u, N)
		# self.add_dynamics_constraint(prog, x, u, N, x_r, u_r, x_current, T)
		self.add_cost(prog, x-x_r.reshape((1,self.nx)), u, N)
		# T = self.add_collision_constraint(prog, x, x_current, other_x_current, T)
		self.add_dynamics_constraint(prog, x, u, N, x_r, u_r, x_current, T)
		self.add_collision_cost(prog, x, other_x_current)

		# Solve the QP
		# solver = OsqpSolver() 
		solver = SnoptSolver()
		result = solver.Solve(prog)

		u_mpc = np.zeros(2) # v and theta_dot

		u_mpc = result.GetSolution(u)[0]
		# x_mpc = result.GetSolution(x)
		# print("Predicted states: ", x_mpc)
		
		return u_mpc

	def compute_lqr_feedback(self, x):
		'''
		Infinite horizon LQR controller
		'''
		A, B = self.continuous_time_linearized_dynamics()
		S = solve_continuous_are(A, B, self.Q, self.R)
		K = -inv(self.R) @ B.T @ S
		u = K @ x
		return u
