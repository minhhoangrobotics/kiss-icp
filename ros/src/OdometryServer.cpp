// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <sophus/se3.hpp>
#include <utility>
#include <vector>

// KISS-ICP-ROS
#include "OdometryServer.hpp"
#include "Utils.hpp"

// KISS-ICP
#include "kiss_icp/pipeline/KissICP.hpp"

// ROS 2 headers
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_srvs/srv/empty.hpp>
#include <tf2_ros/static_transform_broadcaster.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

namespace kiss_icp_ros {

using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;

OdometryServer::OdometryServer(const rclcpp::NodeOptions &options)
    : rclcpp::Node("kiss_icp_node", options) {
    kiss_icp::pipeline::KISSConfig config;
    initializeParameters(config);

    // Construct the main KISS-ICP odometry node
    kiss_icp_ = std::make_unique<kiss_icp::pipeline::KissICP>(config);

    // Initialize subscribers
    pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        // ICP can take longer than one scan period on the simulation host.
        // Keep the freshest scan instead of feeding stale scans to the graph.
        "pointcloud_topic", rclcpp::SensorDataQoS().keep_last(1),
        std::bind(&OdometryServer::RegisterFrame, this, std::placeholders::_1));
    use_leg_prediction_ = declare_parameter<bool>("use_leg_prediction", false);
    use_leg_height_constraint_ = declare_parameter<bool>("use_leg_height_constraint", false);
    RCLCPP_INFO(get_logger(), "Leg motion prediction for KISS ICP: %s",
                use_leg_prediction_ ? "enabled" : "disabled");
    if (use_leg_prediction_ || use_leg_height_constraint_) {
        leg_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "leg_odometry_topic", 20, [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
                recent_leg_poses_.push_back(msg);
                const double newest = rclcpp::Time(msg->header.stamp).seconds();
                while (!recent_leg_poses_.empty() &&
                       newest - rclcpp::Time(recent_leg_poses_.front()->header.stamp).seconds() > 3.0) {
                    recent_leg_poses_.pop_front();
                }
            });
    }
    use_imu_prediction_ = declare_parameter<bool>("use_imu_prediction", false);
    if (use_imu_prediction_) {
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "imu_topic", rclcpp::SensorDataQoS().keep_last(50),
            [this](const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
                recent_imu_.push_back(msg);
                const auto newest = rclcpp::Time(msg->header.stamp);
                while (!recent_imu_.empty() &&
                       (newest - rclcpp::Time(recent_imu_.front()->header.stamp)).seconds() > 2.0) {
                    recent_imu_.pop_front();
                }
            });
    }
    RCLCPP_INFO(get_logger(), "IMU rotation prediction for KISS ICP: %s",
                use_imu_prediction_ ? "enabled" : "disabled");

    // Initialize publishers
    rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("kiss/odometry", qos);
    registration_status_publisher_ = create_publisher<std_msgs::msg::Bool>(
        "kiss/registration_accepted", qos);
    quality_publisher_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        "kiss/registration_quality", qos);
    if (publish_debug_clouds_) {
        frame_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("kiss/frame", qos);
        kpoints_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("kiss/keypoints", qos);
        map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("kiss/local_map", qos);
    }

    // Initialize the transform broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf2_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf2_buffer_->setUsingDedicatedThread(true);
    tf2_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf2_buffer_);
    // Initialize service servers
    reset_service_ = create_service<std_srvs::srv::Empty>(
        "kiss/reset", std::bind(&OdometryServer::ResetService, this, std::placeholders::_1,
                                std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "KISS-ICP ROS 2 odometry node initialized");
}

void OdometryServer::initializeParameters(kiss_icp::pipeline::KISSConfig &config) {
    RCLCPP_INFO(this->get_logger(), "Initializing parameters");

    base_frame_ = declare_parameter<std::string>("base_frame", base_frame_);
    RCLCPP_INFO(this->get_logger(), "\tBase frame: %s", base_frame_.c_str());
    lidar_odom_frame_ = declare_parameter<std::string>("lidar_odom_frame", lidar_odom_frame_);
    RCLCPP_INFO(this->get_logger(), "\tLiDAR odometry frame: %s", lidar_odom_frame_.c_str());
    publish_odom_tf_ = declare_parameter<bool>("publish_odom_tf", publish_odom_tf_);
    RCLCPP_INFO(this->get_logger(), "\tPublish odometry transform: %d", publish_odom_tf_);
    invert_odom_tf_ = declare_parameter<bool>("invert_odom_tf", invert_odom_tf_);
    RCLCPP_INFO(this->get_logger(), "\tInvert odometry transform: %d", invert_odom_tf_);
    publish_debug_clouds_ = declare_parameter<bool>("publish_debug_clouds", publish_debug_clouds_);
    RCLCPP_INFO(this->get_logger(), "\tPublish debug clouds: %d", publish_debug_clouds_);
    debug_map_max_points_ = declare_parameter<int>("debug_map_max_points", 5000);
    position_covariance_ = declare_parameter<double>("position_covariance", 0.1);
    RCLCPP_INFO(this->get_logger(), "\tPosition covariance: %.2f", position_covariance_);
    orientation_covariance_ = declare_parameter<double>("orientation_covariance", 0.1);
    RCLCPP_INFO(this->get_logger(), "\tOrientation covariance: %.2f", orientation_covariance_);

    config.max_range = declare_parameter<double>("data.max_range", config.max_range);
    RCLCPP_INFO(this->get_logger(), "\tMax range: %.2f", config.max_range);
    config.min_range = declare_parameter<double>("data.min_range", config.min_range);
    RCLCPP_INFO(this->get_logger(), "\tMin range: %.2f", config.min_range);
    config.deskew = declare_parameter<bool>("data.deskew", config.deskew);
    RCLCPP_INFO(this->get_logger(), "\tDeskew: %d", config.deskew);
    config.voxel_size = declare_parameter<double>("mapping.voxel_size", config.max_range / 100.0);
    RCLCPP_INFO(this->get_logger(), "\tVoxel size: %.2f", config.voxel_size);
    config.max_points_per_voxel =
        declare_parameter<int>("mapping.max_points_per_voxel", config.max_points_per_voxel);
    RCLCPP_INFO(this->get_logger(), "\tMax points per voxel: %d", config.max_points_per_voxel);
    config.initial_threshold =
        declare_parameter<double>("adaptive_threshold.initial_threshold", config.initial_threshold);
    RCLCPP_INFO(this->get_logger(), "\tInitial threshold: %.2f", config.initial_threshold);
    config.min_motion_th =
        declare_parameter<double>("adaptive_threshold.min_motion_th", config.min_motion_th);
    RCLCPP_INFO(this->get_logger(), "\tMin motion threshold: %.2f", config.min_motion_th);
    config.max_num_iterations =
        declare_parameter<int>("registration.max_num_iterations", config.max_num_iterations);
    RCLCPP_INFO(this->get_logger(), "\tMax number of iterations: %d", config.max_num_iterations);
    config.convergence_criterion = declare_parameter<double>("registration.convergence_criterion",
                                                             config.convergence_criterion);
    RCLCPP_INFO(this->get_logger(), "\tConvergence criterion: %.2f", config.convergence_criterion);
    config.max_num_threads =
        declare_parameter<int>("registration.max_num_threads", config.max_num_threads);
    RCLCPP_INFO(this->get_logger(), "\tMax number of threads: %d", config.max_num_threads);
    config.quality_gate = declare_parameter<bool>("quality.enabled", config.quality_gate);
    config.min_inlier_ratio = declare_parameter<double>("quality.min_inlier_ratio", config.min_inlier_ratio);
    config.max_inlier_rmse = declare_parameter<double>("quality.max_inlier_rmse", config.max_inlier_rmse);
    config.inlier_distance = declare_parameter<double>("quality.inlier_distance", config.inlier_distance);
    config.max_prior_translation = declare_parameter<double>("quality.max_prior_translation", config.max_prior_translation);
    config.max_prior_rotation = declare_parameter<double>("quality.max_prior_rotation", config.max_prior_rotation);
    config.max_prior_vertical = declare_parameter<double>("quality.max_prior_vertical", config.max_prior_vertical);
    if (config.min_inlier_ratio < 0.0 || config.min_inlier_ratio > 1.0 ||
        config.max_inlier_rmse <= 0.0 || config.inlier_distance <= 0.0 ||
        config.max_prior_translation <= 0.0 || config.max_prior_rotation <= 0.0 ||
        config.max_prior_vertical <= 0.0) {
        throw std::invalid_argument("Invalid KISS quality gate parameters");
    }
    if (config.max_range < config.min_range) {
        RCLCPP_WARN(get_logger(),
                    "[WARNING] max_range is smaller than min_range, setting min_range to 0.0");
        config.min_range = 0.0;
    }
}

void OdometryServer::RegisterFrame(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    Sophus::SE3d base_to_lidar;
    if (!base_frame_.empty() && base_frame_ != msg->header.frame_id) {
        try {
            const auto tf = tf2_buffer_->lookupTransform(
                base_frame_, msg->header.frame_id, rclcpp::Time(msg->header.stamp),
                rclcpp::Duration::from_seconds(0.06));
            base_to_lidar = tf2::transformToSophus(tf);
        } catch (const tf2::TransformException &error) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                 "Waiting for LiDAR TF at scan stamp: %s", error.what());
            return;
        }
    }
    const auto leg_pose = (use_leg_prediction_ || use_leg_height_constraint_) ?
                          LegPoseAt(rclcpp::Time(msg->header.stamp)) :
                                            std::optional<Sophus::SE3d>{};
    if (use_leg_prediction_ && !leg_pose) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "No leg odometry near LiDAR timestamp; KISS uses its own motion estimate");
    }
    const auto imu_orientation = use_imu_prediction_ ? ImuBaseOrientationAt(rclcpp::Time(msg->header.stamp)) :
                                                        std::optional<Eigen::Quaterniond>{};
    if (use_imu_prediction_ && !imu_orientation) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "No scan-time IMU attitude within 40 ms; KISS cannot anchor tilt for this cloud");
    }
    if (use_leg_height_constraint_ && !leg_pose) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "No leg odometry within 200 ms of LiDAR; KISS height constraint unavailable");
    }
    std::optional<Sophus::SE3d> pose_hint;
    if (previous_base_to_lidar_) {
        // KISS predicts translation from its last accepted LiDAR motion. Leg
        // odometry can replace that translation, but IMU owns the rotation.
        Sophus::SE3d base_delta = *previous_base_to_lidar_ * kiss_icp_->delta() *
                                  base_to_lidar.inverse();
        if (use_leg_prediction_ && leg_pose && previous_leg_pose_) {
            const auto leg_delta = previous_leg_pose_->inverse() * *leg_pose;
            if (leg_delta.translation().norm() < 0.8) {
                base_delta.translation() = leg_delta.translation();
            }
        }
        if (imu_orientation && previous_imu_base_orientation_ &&
            previous_scan_stamp_.nanoseconds() != 0 &&
            (rclcpp::Time(msg->header.stamp) - previous_scan_stamp_).seconds() < 0.5) {
            const Eigen::Quaterniond delta = previous_imu_base_orientation_->inverse() *
                                              *imu_orientation;
            if (Eigen::AngleAxisd(delta).angle() < 1.0) {
                base_delta = Sophus::SE3d(Sophus::SO3d(delta), base_delta.translation());
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 10000,
                                     "KISS prediction: IMU rotation and %s translation",
                                     use_leg_prediction_ && leg_pose && previous_leg_pose_ ? "leg" : "LiDAR");
            }
        }
        if (base_delta.translation().allFinite() && base_delta.translation().norm() < 0.8) {
            pose_hint = kiss_icp_->pose() * previous_base_to_lidar_->inverse() *
                        base_delta * base_to_lidar;
            if (imu_orientation && odom_from_imu_orientation_) {
                // An incremental IMU delta alone preserves any accumulated
                // ICP roll/pitch error. Re-anchor every prediction to the
                // absolute scan-time IMU attitude instead.
                const Eigen::Quaterniond odom_base =
                    (*odom_from_imu_orientation_ * *imu_orientation).normalized();
                pose_hint = Sophus::SE3d(
                    Sophus::SO3d(
                        (odom_base * base_to_lidar.so3().unit_quaternion()).normalized()),
                    pose_hint->translation());
            }
            if (use_leg_height_constraint_ && leg_pose &&
                odom_base_height_at_leg_anchor_ && leg_height_anchor_) {
                const Sophus::SE3d predicted_base = *pose_hint * base_to_lidar.inverse();
                Eigen::Vector3d t = pose_hint->translation();
                t.z() += (*odom_base_height_at_leg_anchor_ +
                          leg_pose->translation().z() - *leg_height_anchor_) -
                         predicted_base.translation().z();
                pose_hint->translation() = t;
            }
        }
    }
    const auto points = PointCloud2ToEigen(msg);
    const auto timestamps = GetTimestamps(msg);

    // Register frame, main entry point to KISS-ICP pipeline
    const auto &[frame, keypoints] = kiss_icp_->RegisterFrame(points, timestamps, pose_hint);
    const auto &quality = kiss_icp_->registration_quality();
    std_msgs::msg::Bool accepted_msg;
    accepted_msg.data = quality.accepted;
    registration_status_publisher_->publish(accepted_msg);
    std_msgs::msg::Float32MultiArray quality_msg;
    quality_msg.data = {static_cast<float>(quality.inlier_ratio),
                        static_cast<float>(quality.rmse),
                        static_cast<float>(quality.translation_deviation),
                        static_cast<float>(quality.rotation_deviation)};
    quality_publisher_->publish(quality_msg);
    if (kiss_icp_->registration_accepted()) {
      previous_scan_stamp_ = rclcpp::Time(msg->header.stamp);
      previous_base_to_lidar_ = base_to_lidar;
      previous_imu_base_orientation_ = imu_orientation;
      if (imu_orientation && !odom_from_imu_orientation_) {
          const Eigen::Quaterniond odom_base =
              (kiss_icp_->pose() * base_to_lidar.inverse()).so3().unit_quaternion();
          odom_from_imu_orientation_ =
              (odom_base * imu_orientation->conjugate()).normalized();
      }
      if (use_leg_height_constraint_ && leg_pose && !leg_height_anchor_) {
          leg_height_anchor_ = leg_pose->translation().z();
          odom_base_height_at_leg_anchor_ =
              (kiss_icp_->pose() * base_to_lidar.inverse()).translation().z();
      }
    }
    if (leg_pose && kiss_icp_->registration_accepted()) {
        previous_leg_pose_ = *leg_pose;
    } else if (!leg_pose) {
        previous_leg_pose_.reset();
    }

    if (!kiss_icp_->registration_accepted()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "KISS rejected scan (%s): inliers %.2f, RMSE %.3f m, prior deviation %.3f m / %.3f rad, height %.3f m",
                             quality.rejection_reason.c_str(),
                             quality.inlier_ratio, quality.rmse, quality.translation_deviation,
                             quality.rotation_deviation, quality.vertical_deviation);
        return;  // Do not send a rejected LiDAR factor to Graph MSF.
    }

    // Extract the last KISS-ICP pose, ego-centric to the LiDAR
    const Sophus::SE3d kiss_pose = kiss_icp_->pose();

    // Spit the current estimated pose to ROS msgs handling the desired target frame
    PublishOdometry(kiss_pose, msg->header, base_to_lidar);
    // Publishing these clouds is a bit costly, so do it only if we are debugging
    if (publish_debug_clouds_) {
        PublishClouds(frame, keypoints, msg->header);
    }
}

std::optional<Sophus::SE3d> OdometryServer::LegPoseAt(const rclcpp::Time &stamp) const {
    double nearest_error = 0.2;
    std::optional<Sophus::SE3d> result;
    for (const auto &item : recent_leg_poses_) {
        const double error = std::abs((rclcpp::Time(item->header.stamp) - stamp).seconds());
        if (error >= nearest_error) continue;
        const auto &p = item->pose.pose.position;
        const auto &q = item->pose.pose.orientation;
        const Eigen::Quaterniond quaternion(q.w, q.x, q.y, q.z);
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !std::isfinite(quaternion.squaredNorm()) || quaternion.squaredNorm() < 0.5) continue;
        nearest_error = error;
        result = Sophus::SE3d(quaternion.normalized(), Eigen::Vector3d(p.x, p.y, p.z));
    }
    return result;
}

std::optional<Eigen::Quaterniond> OdometryServer::ImuBaseOrientationAt(const rclcpp::Time &stamp) {
    if (recent_imu_.empty()) return std::nullopt;
    sensor_msgs::msg::Imu::ConstSharedPtr before, after;
    for (const auto &item : recent_imu_) {
        if (rclcpp::Time(item->header.stamp) <= stamp) before = item;
        else { after = item; break; }
    }
    const auto sample = before ? before : after;
    if (!sample || std::abs((stamp - rclcpp::Time(sample->header.stamp)).seconds()) > 0.04 ||
        sample->header.frame_id.empty()) return std::nullopt;
    auto from_msg = [](const sensor_msgs::msg::Imu::ConstSharedPtr &item) {
        const auto &q = item->orientation;
        return Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized();
    };
    Eigen::Quaterniond world_imu = from_msg(sample);
    if (before && after) {
        const auto dt = (rclcpp::Time(after->header.stamp) -
                         rclcpp::Time(before->header.stamp)).seconds();
        if (dt > 0.0 && dt < 0.04) {
            const double ratio = (stamp - rclcpp::Time(before->header.stamp)).seconds() / dt;
            world_imu = from_msg(before).slerp(ratio, from_msg(after));
        }
    }
    if (!world_imu.coeffs().allFinite() || world_imu.norm() < 0.5) return std::nullopt;
    if (!base_to_imu_rotation_ || cached_imu_frame_ != sample->header.frame_id) {
        try {
            // The G1 IMU is rigidly attached to pelvis/base_link. Cache the
            // extrinsic so a late joint TF cannot discard a valid IMU sample.
            const auto tf = tf2_buffer_->lookupTransform(
                base_frame_, sample->header.frame_id, tf2::TimePointZero);
            const auto &q = tf.transform.rotation;
            const Eigen::Quaterniond base_imu(q.w, q.x, q.y, q.z);
            if (!base_imu.coeffs().allFinite() || base_imu.norm() < 0.5) return std::nullopt;
            base_to_imu_rotation_ = base_imu.normalized();
            cached_imu_frame_ = sample->header.frame_id;
        } catch (const tf2::TransformException &) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                 "IMU extrinsic TF unavailable; KISS uses LiDAR motion prediction");
            return std::nullopt;
        }
    }
    return (world_imu * base_to_imu_rotation_->inverse()).normalized();
}

void OdometryServer::PublishOdometry(const Sophus::SE3d &kiss_pose,
                                     const std_msgs::msg::Header &header,
                                     const Sophus::SE3d &base_to_lidar) {
    // If necessary, transform the ego-centric pose to the specified base_link/base_footprint frame
    const auto cloud_frame_id = header.frame_id;
    const auto egocentric_estimation = (base_frame_.empty() || base_frame_ == cloud_frame_id);
    const auto moving_frame = egocentric_estimation ? cloud_frame_id : base_frame_;
    Sophus::SE3d pose = kiss_pose;
    if (!egocentric_estimation) {
        if (!initial_base_to_lidar_) initial_base_to_lidar_ = base_to_lidar;
        pose = *initial_base_to_lidar_ * kiss_pose * base_to_lidar.inverse();
    }

    // Broadcast the tf ---
    if (publish_odom_tf_) {
        geometry_msgs::msg::TransformStamped transform_msg;
        transform_msg.header.stamp = header.stamp;
        if (invert_odom_tf_) {
            transform_msg.header.frame_id = moving_frame;
            transform_msg.child_frame_id = lidar_odom_frame_;
            transform_msg.transform = tf2::sophusToTransform(pose.inverse());
        } else {
            transform_msg.header.frame_id = lidar_odom_frame_;
            transform_msg.child_frame_id = moving_frame;
            transform_msg.transform = tf2::sophusToTransform(pose);
        }
        tf_broadcaster_->sendTransform(transform_msg);
    }

    // publish odometry msg
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = header.stamp;
    odom_msg.header.frame_id = lidar_odom_frame_;
    odom_msg.child_frame_id = moving_frame;
    odom_msg.pose.pose = tf2::sophusToPose(pose);
    odom_msg.pose.covariance.fill(0.0);
    odom_msg.pose.covariance[0] = position_covariance_;
    odom_msg.pose.covariance[7] = position_covariance_;
    odom_msg.pose.covariance[14] = position_covariance_;
    odom_msg.pose.covariance[21] = orientation_covariance_;
    odom_msg.pose.covariance[28] = orientation_covariance_;
    odom_msg.pose.covariance[35] = orientation_covariance_;
    odom_publisher_->publish(std::move(odom_msg));
}

void OdometryServer::PublishClouds(const std::vector<Eigen::Vector3d> &frame,
                                   const std::vector<Eigen::Vector3d> &keypoints,
                                   const std_msgs::msg::Header &header) {
    frame_publisher_->publish(std::move(EigenToPointCloud2(frame, header)));
    kpoints_publisher_->publish(std::move(EigenToPointCloud2(keypoints, header)));
    if (map_publisher_->get_subscription_count() == 0) return;
    const auto kiss_map = kiss_icp_->LocalMap();
    std::vector<Eigen::Vector3d> display_map;
    const size_t limit = static_cast<size_t>(std::max(1, debug_map_max_points_));
    const size_t stride = std::max<size_t>(1, (kiss_map.size() + limit - 1) / limit);
    display_map.reserve(std::min(kiss_map.size(), limit));
    for (size_t i = 0; i < kiss_map.size(); i += stride) display_map.push_back(kiss_map[i]);
    auto local_map_header = header;
    local_map_header.frame_id = lidar_odom_frame_;
    map_publisher_->publish(std::move(EigenToPointCloud2(display_map, local_map_header)));
}
void OdometryServer::ResetService(
    [[maybe_unused]] const std::shared_ptr<std_srvs::srv::Empty::Request> request,
    [[maybe_unused]] std::shared_ptr<std_srvs::srv::Empty::Response> response) {
    RCLCPP_INFO(this->get_logger(), "Resetting KISS-ICP map and odometry");

    // Reset the KISS-ICP pipeline
    kiss_icp_->Reset();
    initial_base_to_lidar_.reset();
    previous_leg_pose_.reset();
    previous_base_to_lidar_.reset();
    previous_imu_base_orientation_.reset();
    odom_from_imu_orientation_.reset();
    odom_base_height_at_leg_anchor_.reset();
    leg_height_anchor_.reset();
    previous_scan_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

    RCLCPP_INFO(this->get_logger(), "KISS-ICP reset completed successfully");
}
}  // namespace kiss_icp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(kiss_icp_ros::OdometryServer)
