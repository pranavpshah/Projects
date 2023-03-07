#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

file_path = "path_"
file_waypoints = "waypoints_"
x1 = np.load(file_path+"1.npy")
w1 = np.load(file_waypoints+"1.npy")
x2 = np.load(file_path+"2.npy")
w2 = np.load(file_waypoints+"2.npy")

fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(10)

plt.rcParams.update({'font.size': 22})

plt.plot(x1[:, 0], x1[:, 1], "b-", label="Robot 1 MPC path")
plt.plot(w1[:, 0], w1[: ,1], "rx", label="Robot 1 A* Waypoints")

plt.plot(x2[:, 0], x2[:, 1], "m-", label="Robot 2 MPC path")
plt.plot(w2[:, 0], w2[: ,1], "gx", label="Robot 2 A* Waypoints")

ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("y", fontsize=20)
ax.set_title("MPC paths", fontsize=20)
plt.legend(loc='best', prop={'size': 15})
ax.grid()

file_path = "path_combined"
plt.savefig(file_path+".png")