#include <bits/stdc++.h>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "meam517_interfaces/srv/waypoints.hpp"
#include "geometry_msgs/msg/point.hpp"

#include <memory>

using namespace std;


// TODO: Add ROS
// TODO: ROS waypoint message should contain x,y,theta

typedef pair <int, int> xy;
vector<vector<int>> grid
{
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0}
}; 
// 0 - Free
// 1 - Occupied

struct waypoint{
    float cost;
    xy location;
    xy parent;
};

struct CompareCost {
    bool operator()(waypoint const& w1, waypoint const& w2)
    {
        return w1.cost > w2.cost;
    }
};


vector<xy> get_neighbors(int x, int y)
{
    vector<xy> neighbors;
    if(x!=0){
        neighbors.push_back(make_pair(x-1, y)); // West
        if(y!=0)
            neighbors.push_back(make_pair(x-1, y-1)); // North west
        if(y!=grid[0].size()-1)
            neighbors.push_back(make_pair(x-1, y+1)); // South west
    }
    if(x!=grid.size()-1){
        neighbors.push_back(make_pair(x+1, y)); // East
        if(y!=0)
            neighbors.push_back(make_pair(x+1, y-1)); // North east
        if(y!=grid[0].size()-1)
            neighbors.push_back(make_pair(x+1, y+1)); // South east
    }
    if(y!=0)
        neighbors.push_back(make_pair(x, y-1)); // North
    if(y!=grid[0].size()-1)
        neighbors.push_back(make_pair(x, y+1)); // South
        
    return neighbors;
}

float get_distance(int x1, int y1, int x2, int y2)
{
    return sqrt(
        pow(x1-x2, 2) + pow(y1-y2, 2)
    );
}

vector<xy> astar(xy start, xy goal, vector<vector<int>> grid)
{
    cout << "A* called" << endl;
    vector<xy> path;
    priority_queue<waypoint, vector<waypoint>, CompareCost> minheap;
    set<xy> visited;
    vector<vector<float>> dist(grid.size(), vector<float>(grid[0].size(), INFINITY));
    vector<vector<xy>> parent(grid.size(), vector<xy>(grid[0].size()));

    waypoint s = {0, start, make_pair(-1,-1)};
    dist[start.first][start.second] = 0;
    minheap.push(s);
    parent[start.first][start.second] = make_pair(-1, -1);
    
    while(!minheap.empty()){
        waypoint top = minheap.top();
        // cout << top.location.first << ", " << top.location.second << endl;
        minheap.pop();
        // cout << visited.count(top.location) << endl;
        if(visited.count(top.location)==0){
            visited.insert(top.location);
            parent[top.location.first][top.location.second] = top.parent;
            if(top.location == goal){
                // GOAL condition
                xy path_next = top.location;
                while(path_next != make_pair(-1,-1)){
                    path.push_back(path_next);
                    path_next = parent[path_next.first][path_next.second];
                }
                reverse(path.begin(), path.end());
                
                return path;
            }


            for(auto n : get_neighbors(top.location.first, top.location.second)){
                if(grid[n.first][n.second]==0){
                    float f = dist[n.first][n.second];
                    float g = top.cost + 
                            get_distance(top.location.first, top.location.second, n.first, n.second) +
                            get_distance(goal.first, goal.second, n.first, n.second);
                    
                    if(g<f){
                        dist[top.location.first][top.location.second] = g;
                        waypoint neighbor = {g, n, top.location};
                        minheap.push(neighbor);
                    }
                }
            }
        }
    }

}

void find_path(const std::shared_ptr<meam517_interfaces::srv::Waypoints::Request> request,
          std::shared_ptr<meam517_interfaces::srv::Waypoints::Response>     response)
{
    cout << "Entered service" << endl;
    xy start, goal;
    start.first = request->start.x;
    start.second = request->start.y;
    goal.first = request->end.x;
    goal.second = request->end.y;

    cout << "Start: " << start.first << ", " << start.second << endl;
    cout << "Goal: " << goal.first << ", " << goal.second << endl;

    vector<xy> path = astar(start, goal, grid);
    // for(auto p : path){
    //     cout << p.first << ", " << p.second << endl;
    // }
    for(auto p : path){
        geometry_msgs::msg::Point point;
        point.x = p.first;
        point.y = p.second;
        response.get()->path.push_back(point);
    }
    // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "path: %d", path.size());
    // for(auto p : response->path){
    //     cout << p.x << ", " << p.y << endl;
    // }
}

int main(int argc, const char** argv)
{
    rclcpp::init(argc, argv);

    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("find_path_server");

    rclcpp::Service<meam517_interfaces::srv::Waypoints>::SharedPtr service =
        node->create_service<meam517_interfaces::srv::Waypoints>("find_path", &find_path);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Ready to find path");

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}