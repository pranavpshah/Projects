
## Chance Constrained Multi-Agent Non-linear MPC

A non-linear Model Predictive Controller (MPC) for multi agent systems

#### Chance-constrained
Constraints on the probablilty of a collision between the robots. The probability of collision is directly proportional to the area of overlap of the uncertain robot regions.

:link: [Poster](https://drive.google.com/file/d/1UIUwToIc4GugEmIqlzthz3QaQL_ka6ON/view?usp=sharing)

:link: [Report](https://drive.google.com/file/d/1m9zmTFQniYrAPJRiMssyLQJa_hi_x13q/view?usp=sharing)

:link: [Plots and Videos](https://drive.google.com/drive/folders/1IphhiGbgVlrLbb6JlH6qwuUXd3AbcMIj?usp=sharing)




#### Instructions:
Unzip the turtlebot3_simulation zip file and add it to the gitignore list. 


So, to run the Gazebo Simulator, one path will need to be manually added.
After pulling the Repo, open the file: MEAM-517-Project/src/robot_spawner_pkg/src/robot_spawner_pkg/spawn_turtlebot.py

Change the below line :
```
sdf_file_path = "/home/aadith/Desktop/MEAM-517-Project/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model-1_4.sdf"
```

to reflect the path to whereever you have the REPO
```
sdf_file_path = "--ADD PATH TO REPO--/MEAM-517-Project/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model-1_4.sdf"
```

After pulling the Repo, open the file: MEAM-517-Project/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model-1_4.sdf

There are four instances of hardcoded path, change them to reflect your path!


It would be nice if we could get our Robot to take this model path in one location, but I don't know how it works


#### Build instructions:
1. Navigate to root of the worksopace
2. Run 
     `$ colcon build --packages-up-to meam517_interfaces`
3. Run 
     `$ source install/setup.bash`
4. Run 
     `$ colcon build`
5. Run 
     `$ source install/setup.bash`


To launch simulator with multiple turtlebots, Run:

```bash multi_sim_launch.sh <num_robots>```

To Control robot with teleop:

``` sh teleop.sh <robot number> ```

To launch MPC controller for a given robot run:

```ros2 run mpc single_mpc_control.py <total number of robots> <robot ID> ```
