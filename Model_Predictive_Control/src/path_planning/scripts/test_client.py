#!/usr/bin/env python3
import sys

from meam517_interfaces.srv import Waypoints
from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node


class ClientAsync(Node):

    def __init__(self):
        super().__init__('astar_python_client')
        self.cli = self.create_client(Waypoints, 'find_path')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Waypoints.Request()

    def send_request(self, xs, ys, xg, yg):
        self.req.start.x = xs
        self.req.start.y = ys
        self.req.end.x = xg
        self.req.end.y = yg
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    minimal_client = ClientAsync()
    response = minimal_client.send_request(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    size = len(response.path)
    for i in range(size):
        print(response.path[i].x, response.path[i].y)

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()