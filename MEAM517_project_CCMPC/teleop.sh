
export TURTLEBOT3_MODEL="burger"
ros2 run turtlebot3_teleop teleop_keyboard --ros-args --remap cmd_vel:=robot${1}/cmd_vel

