
MEAM 517 Project Readme!

<!-- :link: [Google Drive](https://drive.google.com/drive/folders/12vvI-4S0ICZvCfdP6TQElmgLZ1OQ7fW2?usp=sharing) -->


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

build instructions:
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

     
     
     
   
