#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "OdometryServer.hpp"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    auto node = std::make_shared<kiss_icp_ros::OdometryServer>(rclcpp::NodeOptions());
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
