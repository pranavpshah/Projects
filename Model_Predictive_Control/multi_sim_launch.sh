#!/bin/bash

export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/home/rithwik/UPenn/MEAM517/MEAM-517-Project/src/turtlebot3_simulations/turtlebot3_gazebo/models;


gazebo --verbose -s libgazebo_ros_factory.so &

# i=1
# dist=1
# while [[ $i -le $1 ]]
# do
#  ros2 run robot_spawner_pkg spawn_turtlebot robot$i robot$i $dist 0.0 0.1;
#  echo "Hello$i"
#  ((dist = dist - 2))
#  ((i = i + 1))
# done

ros2 run robot_spawner_pkg spawn_turtlebot robot1 robot1 0.0 0.0 0.1;
ros2 run robot_spawner_pkg spawn_turtlebot robot2 robot2 5.0 0.0 0.1;
ros2 run robot_spawner_pkg spawn_turtlebot robot3 robot3 2.0 0.0 0.1;

#ros2 run mpc sample_state_update_code.py
