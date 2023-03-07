# MEAM620 - An Autonomous VIO-based Quadcoptor

Controller :- Non-linear geometric controller
Path planning :- A* algorithm
Trajectory planning :- Minimum-jerk trajectory
State estimation is done by combining IMU data with image data using a Complementary filter.

## How to run
Got to file `./proj3/code/sandbox.py`
Run `sandbox.py` to the simulation animation and plots. 
A* implemented in file `graph_search.py`, minimum jerk trajetory implemented in `world_traj.py`, geometric controller implemented in `se3_control.py` and VIO algorithm implemented in `vio.py`.