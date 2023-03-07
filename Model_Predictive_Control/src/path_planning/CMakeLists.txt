cmake_minimum_required(VERSION 3.8)
project(path_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(meam517_interfaces REQUIRED)


add_executable(cpp_publisher scripts/cpp_pub.cpp)
add_executable(find_path_server scripts/astar.cpp)
ament_target_dependencies(find_path_server rclcpp std_msgs geometry_msgs meam517_interfaces)
add_executable(find_path_client scripts/test_waypoint_client.cpp)
ament_target_dependencies(find_path_client rclcpp std_msgs geometry_msgs meam517_interfaces)

install(TARGETS
  cpp_publisher
  find_path_server
  find_path_client
  DESTINATION lib/${PROJECT_NAME})


# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/python_subscriber.py
  scripts/test_client.py
  DESTINATION lib/${PROJECT_NAME}
)




if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
