// SPDX-License-Identifier: MIT
#pragma once
#include <optional>
#include <sophus/se3.hpp>

namespace kiss_icp {
// Mount transforms are T_base_lidar at their respective scan timestamps.
// relative_base_motion must come from the two previous accepted base poses.
inline Sophus::SE3d PredictLidarPose(
    const Sophus::SE3d& previous_lidar_pose, const Sophus::SE3d& previous_mount,
    const Sophus::SE3d& relative_base_motion, const Sophus::SE3d& current_mount,
    const std::optional<Sophus::SO3d>& absolute_base_orientation = std::nullopt) {
    auto base = previous_lidar_pose * previous_mount.inverse() * relative_base_motion;
    if (absolute_base_orientation)
        base = Sophus::SE3d(*absolute_base_orientation, base.translation());
    return base * current_mount;
}
}  // namespace kiss_icp
