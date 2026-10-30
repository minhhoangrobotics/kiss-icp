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
#include "kiss_icp/core/MotionPrediction.hpp"

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
    // Run short sensor callbacks while the LiDAR callback performs ICP. The
    // executable uses a two-thread executor, so scan-time IMU
    // and leg samples can arrive before registration needs them.
    sensor_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions sensor_options;
    sensor_options.callback_group = sensor_callback_group_;
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
                std::lock_guard<std::mutex> lock(sensor_queue_mutex_);
                recent_leg_poses_.push_back(msg);
                const double newest = rclcpp::Time(msg->header.stamp).seconds();
                while (!recent_leg_poses_.empty() &&
                       newest - rclcpp::Time(recent_leg_poses_.front()->header.stamp).seconds() > 3.0) {
                    recent_leg_poses_.pop_front();
                }
            }, sensor_options);
    }
    use_imu_prediction_ = declare_parameter<bool>("use_imu_prediction", false);
    use_inertial_translation_ = declare_parameter<bool>("inertial_prediction.enabled", false);
    inertial_prediction_.acceleration_noise = declare_parameter<double>("inertial_prediction.acceleration_noise", 0.2);
    inertial_prediction_.bias_walk = declare_parameter<double>("inertial_prediction.bias_walk", 0.005);
    inertial_prediction_.position_noise = declare_parameter<double>("inertial_prediction.position_noise", 0.02);
    inertial_prediction_.max_sample_gap = declare_parameter<double>("inertial_prediction.max_sample_gap", 0.05);
    gravity_magnitude_ = declare_parameter<double>("inertial_prediction.gravity_magnitude", 9.806);
    for (double value : {inertial_prediction_.acceleration_noise, inertial_prediction_.bias_walk,
                         inertial_prediction_.position_noise, inertial_prediction_.max_sample_gap,
                         gravity_magnitude_})
        if (!std::isfinite(value) || value<=0.0) throw std::invalid_argument("Invalid inertial_prediction parameter");
    if (use_inertial_translation_ && !use_imu_prediction_)
        throw std::invalid_argument("inertial_prediction requires use_imu_prediction");
    RCLCPP_INFO(get_logger(), "Inertial translation prediction: %s; LiDAR updates observable directions only",
                use_inertial_translation_ ? "enabled" : "disabled");
    if (use_imu_prediction_) {
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "imu_topic", rclcpp::SensorDataQoS().keep_last(50),
            [this](const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
                std::lock_guard<std::mutex> lock(sensor_queue_mutex_);
                recent_imu_.push_back(msg);
                const auto newest = rclcpp::Time(msg->header.stamp);
                while (!recent_imu_.empty() &&
                       (newest - rclcpp::Time(recent_imu_.front()->header.stamp)).seconds() > 2.0) {
                    recent_imu_.pop_front();
                }
            }, sensor_options);
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
    config.fix_rotation_from_hint = declare_parameter<bool>(
        "registration.fix_rotation_from_imu", config.fix_rotation_from_hint);
    config.rotation_prior_information = declare_parameter<double>(
        "registration.rotation_prior_information", config.rotation_prior_information);
    const auto metric = declare_parameter<std::string>("registration.metric", "point_to_point");
    if (metric != "point_to_point" && metric != "point_to_point_exact" &&
        metric != "point_to_plane" && metric != "generalized_icp" && metric != "hybrid")
        throw std::invalid_argument("Unknown registration.metric");
    config.surface.enabled = metric != "point_to_point";
    const double regularization = declare_parameter<double>("registration.covariance_regularization", 0.01);
    if (!std::isfinite(regularization) || regularization <= 0.0 || regularization > 1.0)
        throw std::invalid_argument("registration.covariance_regularization must be in (0,1]");
    config.surface.directional_hybrid = metric == "hybrid";
    config.surface.covariance_regularization = (metric == "point_to_plane" || metric == "hybrid") ? 0.0 :
                                               metric == "point_to_point_exact" ? 1.0 : regularization;
    config.surface.translation_eigenvalue_ratio = declare_parameter<double>(
        "registration.translation_eigenvalue_ratio", 0.02);
    if (!std::isfinite(config.surface.translation_eigenvalue_ratio) ||
        config.surface.translation_eigenvalue_ratio <= 0.0 || config.surface.translation_eigenvalue_ratio >= 1.0)
        throw std::invalid_argument("registration.translation_eigenvalue_ratio must be in (0,1)");
    config.surface.neighbors = declare_parameter<int>("registration.normal_neighbors", 10);
    config.surface.radius = declare_parameter<double>("registration.normal_radius", 1.2);
    config.surface.planarity_ratio = declare_parameter<double>("registration.planarity_ratio", 0.15);
    config.surface.eigenvalue_ratio = declare_parameter<double>("registration.eigenvalue_ratio", 1e-4);
    if (config.surface.neighbors < 3 || !std::isfinite(config.surface.radius) ||
        config.surface.radius <= 0.0 || !std::isfinite(config.surface.planarity_ratio) ||
        config.surface.planarity_ratio <= 0.0 || config.surface.planarity_ratio >= 1.0 ||
        !std::isfinite(config.surface.eigenvalue_ratio) || config.surface.eigenvalue_ratio <= 0.0 ||
        config.surface.eigenvalue_ratio >= 1.0)
        throw std::invalid_argument("Invalid surface registration parameters");
    RCLCPP_INFO(get_logger(), "KISS registration metric: %s; normal K=%d radius=%.2f m; rank ratio=%.1e; covariance epsilon=%.3g",
                metric.c_str(), config.surface.neighbors, config.surface.radius,
                config.surface.eigenvalue_ratio, config.surface.covariance_regularization);
    if (!std::isfinite(config.rotation_prior_information) ||
        config.rotation_prior_information < 0.0) {
        throw std::invalid_argument("registration.rotation_prior_information must be finite and nonnegative");
    }
    config.quality_gate = declare_parameter<bool>("quality.enabled", config.quality_gate);
    config.min_inlier_ratio = declare_parameter<double>("quality.min_inlier_ratio", config.min_inlier_ratio);
    config.max_inlier_rmse = declare_parameter<double>("quality.max_inlier_rmse", config.max_inlier_rmse);
    config.inlier_distance = declare_parameter<double>("quality.inlier_distance", config.inlier_distance);
    config.max_prior_translation = declare_parameter<double>("quality.max_prior_translation", config.max_prior_translation);
    config.max_prior_rotation = declare_parameter<double>("quality.max_prior_rotation", config.max_prior_rotation);
    reseed_after_rejection_seconds_ = declare_parameter<double>(
        "quality.reseed_after_rejection_seconds", reseed_after_rejection_seconds_);
    if (config.min_inlier_ratio < 0.0 || config.min_inlier_ratio > 1.0 ||
        config.max_inlier_rmse <= 0.0 || config.inlier_distance <= 0.0 ||
        config.max_prior_translation <= 0.0 || config.max_prior_rotation <= 0.0 ||
        reseed_after_rejection_seconds_ <= 0.0) {
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
        const double scan_interval =
            (rclcpp::Time(msg->header.stamp) - previous_scan_stamp_).seconds();
        // This delta was measured between the two previous accepted BASE
        // poses, each using its own scan-time mount transform. The current
        // mount must never be used to reconstruct a past motion increment.
        Sophus::SE3d base_delta = previous_base_delta_;
        if (previous_scan_interval_ > 0.0 && scan_interval > 0.0 && scan_interval < 0.5)
            base_delta = Sophus::SE3d::exp(base_delta.log() *
                                          (scan_interval / previous_scan_interval_));
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
            std::optional<Sophus::SO3d> absolute_base_orientation;
            if (imu_orientation && odom_from_imu_orientation_) {
                // An incremental IMU delta alone preserves any accumulated
                // ICP roll/pitch error. Re-anchor every prediction to the
                // absolute scan-time IMU attitude instead.
                const Eigen::Quaterniond odom_base =
                    (*odom_from_imu_orientation_ * *imu_orientation).normalized();
                absolute_base_orientation = Sophus::SO3d(odom_base);
            }
            // Compose the sensor lever arm AFTER choosing the current base
            // attitude, so pure base rotation moves the LiDAR correctly.
            pose_hint = kiss_icp::PredictLidarPose(kiss_icp_->pose(), *previous_base_to_lidar_,
                                                  base_delta, base_to_lidar,
                                                  absolute_base_orientation);
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
    bool inertial_translation_valid = false;
    if (use_inertial_translation_ && inertial_prediction_.initialized() &&
        imu_orientation && odom_from_imu_orientation_) {
        std::vector<kiss_icp::InertialSample> samples;
        {
            std::lock_guard<std::mutex> lock(sensor_queue_mutex_);
            samples.reserve(recent_imu_.size());
            for (const auto& item : recent_imu_) {
                if (item->header.frame_id != cached_imu_frame_) continue;
                const auto& q=item->orientation;
                const auto& a=item->linear_acceleration;
                Eigen::Quaterniond orientation(q.w,q.x,q.y,q.z);
                Eigen::Vector3d acceleration(a.x,a.y,a.z);
                const double time=rclcpp::Time(item->header.stamp).seconds();
                if (!orientation.coeffs().allFinite() || orientation.norm()<0.5 ||
                    !acceleration.allFinite() || (!samples.empty() && time<=samples.back().time)) continue;
                samples.push_back({time, (*odom_from_imu_orientation_*orientation.normalized()).normalized(),
                                   acceleration});
            }
        }
        inertial_translation_valid = inertial_prediction_.Predict(rclcpp::Time(msg->header.stamp).seconds(),samples);
        if (inertial_translation_valid) {
            const Sophus::SO3d rotation((*odom_from_imu_orientation_ * *imu_orientation).normalized());
            const Sophus::SE3d base(rotation,inertial_prediction_.position()-rotation*base_to_imu_translation_);
            pose_hint = base * base_to_lidar;
        } else {
            RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),5000,
                                "Inertial translation unavailable: IMU does not cover the scan interval");
        }
    }
    const auto points = PointCloud2ToEigen(msg);
    const auto timestamps = GetTimestamps(msg);

    // Register frame, main entry point to KISS-ICP pipeline
    auto registration = kiss_icp_->RegisterFrame(points, timestamps, pose_hint,
                                                 imu_orientation.has_value(), inertial_translation_valid);
    const auto &first_quality = kiss_icp_->registration_quality();
    const double rejected_age = previous_scan_stamp_.nanoseconds() == 0 ? 0.0 :
        (rclcpp::Time(msg->header.stamp) - previous_scan_stamp_).seconds();
    if (!first_quality.accepted && use_leg_height_constraint_ && leg_pose && previous_leg_pose_ &&
        previous_base_to_lidar_ && rejected_age >= reseed_after_rejection_seconds_) {
        // Any sustained rejection leaves the local voxel map behind the robot.
        // Rebuild it at the scan-time leg pose, retaining the KISS odom origin.
        // A low inlier ratio is a reason to reseed, not a reason to wait forever.
        const Sophus::SE3d leg_delta = previous_leg_pose_->inverse() * *leg_pose;
        Sophus::SE3d predicted_base =
            kiss_icp_->pose() * previous_base_to_lidar_->inverse() * leg_delta;
        if (imu_orientation && odom_from_imu_orientation_) {
            predicted_base = Sophus::SE3d(
                Sophus::SO3d((*odom_from_imu_orientation_ * *imu_orientation).normalized()),
                predicted_base.translation());
        }
        const Sophus::SE3d predicted_lidar = predicted_base * base_to_lidar;
        RCLCPP_WARN(get_logger(),
                    "Reseeding stale KISS voxel map after %.2f s of rejected scans (%s); "
                    "leg delta %.2f m, previous RMSE %.3f m",
                    rejected_age, first_quality.rejection_reason.c_str(),
                    leg_delta.translation().norm(), first_quality.rmse);
        kiss_icp_->ResetToPose(predicted_lidar);
        registration = kiss_icp_->RegisterFrame(points, timestamps, predicted_lidar,
                                                imu_orientation.has_value());
        // The reseeded map has a new translational reference. Reinitialize
        // the filter at the accepted pose instead of correcting a stale prior.
        inertial_translation_valid = false;
    }
    const auto &[frame, keypoints] = registration;
    const auto &quality = kiss_icp_->registration_quality();
    const auto& surface = kiss_icp_->surface_diagnostics();
    if (surface.correspondences > 0 || surface.fallback) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
            "KISS surface: matches=%zu rank=%d/6 observable_translation=%.0f/3 metric_RMSE=%.4f m inertial=%s fallback=%s",
            surface.correspondences, surface.rank, surface.observable_translation.trace(), surface.residual_rmse,
            inertial_translation_valid ? "yes" : "no",
            surface.fallback ? "point_to_point" : "no");
    }
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
      const Sophus::SE3d current_base_pose = kiss_icp_->pose() * base_to_lidar.inverse();
      if (previous_base_pose_) {
          previous_base_delta_ = previous_base_pose_->inverse() * current_base_pose;
          previous_scan_interval_ =
              (rclcpp::Time(msg->header.stamp) - previous_scan_stamp_).seconds();
      }
      previous_base_pose_ = current_base_pose;
      previous_scan_stamp_ = rclcpp::Time(msg->header.stamp);
      previous_base_to_lidar_ = base_to_lidar;
      previous_imu_base_orientation_ = imu_orientation;
      if (imu_orientation && !odom_from_imu_orientation_) {
          const Eigen::Quaterniond odom_base =
              (kiss_icp_->pose() * base_to_lidar.inverse()).so3().unit_quaternion();
          odom_from_imu_orientation_ =
              (odom_base * imu_orientation->conjugate()).normalized();
      }
      if (use_inertial_translation_ && odom_from_imu_orientation_ && imu_orientation) {
          const Eigen::Vector3d imu_position=current_base_pose.translation()+
                                             current_base_pose.so3()*base_to_imu_translation_;
          if (inertial_translation_valid)
              inertial_prediction_.Correct(imu_position, surface.observable_translation);
          else
              inertial_prediction_.Initialize(rclcpp::Time(msg->header.stamp).seconds(),imu_position,
                  *odom_from_imu_orientation_ * Eigen::Vector3d(0,0,-gravity_magnitude_));
          const int observable=static_cast<int>(std::lround(surface.observable_translation.trace()));
          if(observable!=last_observable_rank_) {
              RCLCPP_INFO(get_logger(),"LiDAR translation observability: %d/3; unobserved motion uses inertial prediction",observable);
              last_observable_rank_=observable;
          }
          RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),5000,
              "Inertial motion: speed %.3f m/s, bias norm %.4f m/s^2, position sigma norm %.3f m",
              inertial_prediction_.velocity().norm(),inertial_prediction_.bias().norm(),
              std::sqrt(std::max(0.0,inertial_prediction_.covariance().topLeftCorner<3,3>().trace())));
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
    std::lock_guard<std::mutex> lock(sensor_queue_mutex_);
    nav_msgs::msg::Odometry::ConstSharedPtr before, after;
    for (const auto &item : recent_leg_poses_) {
        if (rclcpp::Time(item->header.stamp) <= stamp) before = item;
        else { after = item; break; }
    }
    auto valid_pose = [](const nav_msgs::msg::Odometry::ConstSharedPtr &item)
        -> std::optional<Sophus::SE3d> {
        if (!item) return std::nullopt;
        const auto &p = item->pose.pose.position;
        const auto &q = item->pose.pose.orientation;
        const Eigen::Quaterniond quaternion(q.w, q.x, q.y, q.z);
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !std::isfinite(quaternion.squaredNorm()) || quaternion.squaredNorm() < 0.5)
            return std::nullopt;
        return Sophus::SE3d(quaternion.normalized(), Eigen::Vector3d(p.x, p.y, p.z));
    };
    if (before && after) {
        const double dt = (rclcpp::Time(after->header.stamp) -
                           rclcpp::Time(before->header.stamp)).seconds();
        if (dt > 0.0 && dt < 0.04) {
            const auto earlier = valid_pose(before), later = valid_pose(after);
            if (earlier && later) {
                const double u = (stamp - rclcpp::Time(before->header.stamp)).seconds() / dt;
                const Eigen::Quaterniond rotation = earlier->so3().unit_quaternion().slerp(
                    u, later->so3().unit_quaternion());
                const Eigen::Vector3d translation =
                    (1.0 - u) * earlier->translation() + u * later->translation();
                return Sophus::SE3d(rotation.normalized(), translation);
            }
        }
    }
    // A one-sided sample can still be useful during callback scheduling, but
    // avoid extrapolating a fast turn from an old leg pose.
    const auto nearest = before && after ?
        ((stamp - rclcpp::Time(before->header.stamp)) <=
         (rclcpp::Time(after->header.stamp) - stamp) ? before : after) :
        (before ? before : after);
    if (!nearest || std::abs((stamp - rclcpp::Time(nearest->header.stamp)).seconds()) > 0.03)
        return std::nullopt;
    return valid_pose(nearest);
}

std::optional<Eigen::Quaterniond> OdometryServer::ImuBaseOrientationAt(const rclcpp::Time &stamp) {
    sensor_msgs::msg::Imu::ConstSharedPtr before, after;
    {
        std::lock_guard<std::mutex> lock(sensor_queue_mutex_);
        if (recent_imu_.empty()) return std::nullopt;
        for (const auto &item : recent_imu_) {
            if (rclcpp::Time(item->header.stamp) <= stamp) before = item;
            else { after = item; break; }
        }
    }
    auto sample = before ? before : after;
    if (before && after &&
        rclcpp::Time(after->header.stamp) - stamp <
        stamp - rclcpp::Time(before->header.stamp)) {
        sample = after;
    }
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
            const auto& p=tf.transform.translation;
            base_to_imu_translation_=Eigen::Vector3d(p.x,p.y,p.z);
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
    previous_base_pose_.reset();
    previous_base_delta_ = Sophus::SE3d();
    previous_scan_interval_ = 0.0;
    inertial_prediction_.Reset();
    last_observable_rank_ = -1;
    previous_imu_base_orientation_.reset();
    odom_from_imu_orientation_.reset();
    odom_base_height_at_leg_anchor_.reset();
    leg_height_anchor_.reset();
    previous_scan_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    {
        std::lock_guard<std::mutex> lock(sensor_queue_mutex_);
        recent_imu_.clear();
        recent_leg_poses_.clear();
    }

    RCLCPP_INFO(this->get_logger(), "KISS-ICP reset completed successfully");
}
}  // namespace kiss_icp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(kiss_icp_ros::OdometryServer)
