#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, Pose


current_pose =Pose();                                   # Global variable that updates state when required

class PoseSubscriber(Node):                             # Node that houses the subscriber and updates global var for position

    def __init__(self):
        super().__init__('pose_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/robot1/state',                            # The topic to listen to
            self.update_global_state,
            10)
        self.subscription # prevent unused variable warning

    def update_global_state(self, msg):
        global current_pose;
        current_pose = msg.pose.pose;                   # Pose contains position(current_pose.position.x/y/z) and orientation (current_pose.orientation.w/x/y/z)
        #self.get_logger().info("pose updated")



class SamplePublisher(Node):

    def __init__(self):
        super().__init__('sample_publisher')
        self.publisher_ = self.create_publisher(Pose, 'my_pose', 10)


def main(args=None):
    rclpy.init(args=args)
    print("Listening to state update!")
    
    sample_publisher = SamplePublisher()
    pose_subscriber = PoseSubscriber()
    #rclpy.spin(pose_subscriber)

    global current_pose;
    while True:
        """
            TODO : Fill code as needed
            call the spin once function to read for inturrupts each time you want the variable to be updated
        """
        rclpy.spin_once(pose_subscriber);               # Scan and handle interupts if required : i.e, update current_state variable
        
        print("Doing some calculation");                # Run any task 
        print("Calling some functions");
        print("Doing work");

                                                        # Observe that variable has changed
        print("X: ", current_pose.position.x, "   Y: ", current_pose.position.y, "   Z: ", current_pose.position.z)
        sample_publisher.publisher_.publish(current_pose)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pose_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()