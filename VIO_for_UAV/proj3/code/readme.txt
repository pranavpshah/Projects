===============================
MEAM 620 Project 3
Name: Pranav Shah
===============================

graph_search.py
===================================================================================================
Same code as used in Project 1.2.
"node.py" is also used in this code.
===================================================================================================

se3_control.py
===================================================================================================
Gains had to be re-tuned and made to have a more agressive response.
The gains were tuned to have a settling time of around 0.15 s for the attitude controller.
The position controller then was tuned to have an approximate settling time of around an order
higher than the attitude controller.
While tuning the gains, the overshoot was maintained less than 2%.
===================================================================================================

world_traj.py
===================================================================================================
Minimum Jerk trajectory was implemented as explained in lectures.
The average speed for each segment was taken to be 2.9 m/s.
The path obtained from "graph_search.py" was pruned by taking every 7 waypoint and discarding
others.
The time duration for the first and last segment for first and last segment was multiplied by a 
fixed scalar value so as to provide more time to the quad-rotor to come to a stop at the end. 
The time duration for the remaining segments is determined based on a non-linear, sine and cosine
based function.
===================================================================================================

vio.py
===================================================================================================
Same code as used in Project 2.3 to get the state estimates of the quadrotor.
===================================================================================================