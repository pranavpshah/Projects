#include "rclcpp/rclcpp.hpp"
#include "meam517_interfaces/srv/waypoints.hpp"

#include "geometry_msgs/msg/point.hpp"

#include <chrono>
#include <cstdlib>
#include <memory>

using namespace std::chrono_literals;

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    if (argc != 5) {
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "usage: astar_client start_x start_y goal_x goal_y");
        return 1;
    }

    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("find_path_client");
    rclcpp::Client<meam517_interfaces::srv::Waypoints>::SharedPtr client =
        node->create_client<meam517_interfaces::srv::Waypoints>("find_path");

    auto request = std::make_shared<meam517_interfaces::srv::Waypoints::Request>();
    request.get()->start.x = std::stoi(argv[1]);
    request.get()->start.y = std::stoi(argv[2]);
    request.get()->end.x = std::stoi(argv[3]);
    request.get()->end.y = std::stoi(argv[4]);
    
    while (!client->wait_for_service(1s)) {
        if (!rclcpp::ok()) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
        return 0;
        }
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "service not available, waiting again...");
    }

    auto result = client->async_send_request(request);
    // Wait for the result.
    if (rclcpp::spin_until_future_complete(node, result) ==
        rclcpp::FutureReturnCode::SUCCESS)
    {
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Waypoint found");
        std::vector<geometry_msgs::msg::Point, std::allocator<geometry_msgs::msg::Point>> path = result.get()->path;
        int size = path.size();
        std::cout << "PATH:" << std::endl;
        for(int i=0;i<size;i++){
            std::cout << path[i].x << ", " << path[i].y << std::endl;
        }
        
        
        
        
        std::cout << size << std::endl;
        // std::cout << result.get()->path[0].x << std::endl;
        // for(std::vector<geometry_msgs::msg::Point>::const_iterator it = result.get()->path.begin(); it != result.get()->path.end(); ++it){
        //     std::cout << "pls" << std::endl;
        //     std::cout << it->x << ", " << it->y <<std::endl;
        //     // std::cout << result.get()->path.at(1).x << ", " << result.get()->path.at(1).y << std::endl;
        // }
    } else {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service find_path");
    }

    rclcpp::shutdown();
    return 0;
}