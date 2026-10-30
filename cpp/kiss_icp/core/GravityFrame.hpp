// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace kiss_icp {
// Keep the first base position and heading as the odometry origin, but use
// gravity for its vertical axis. An initial body pitch must not rotate all
// future translations into a sloping trajectory in a gravity-aligned backend.
inline Sophus::SE3d GravityAlignedOdometryFrame(const Sophus::SE3d& internal_from_base,
                                               const Eigen::Quaterniond& world_from_base) {
    const Eigen::Matrix3d rotation=world_from_base.normalized().toRotationMatrix();
    const double yaw=std::atan2(rotation(1,0),rotation(0,0));
    const Eigen::Quaterniond level_from_base=
        Eigen::Quaterniond(Eigen::AngleAxisd(-yaw,Eigen::Vector3d::UnitZ()))*world_from_base.normalized();
    return Sophus::SE3d(Sophus::SO3d(level_from_base),Eigen::Vector3d::Zero())*internal_from_base.inverse();
}
}  // namespace kiss_icp
