// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace kiss_icp {
// One-dimensional plane innovation at the independent motion prediction.
// Both supports have voxel-scale spatial uncertainty (r^2/12 each). Attitude
// uncertainty grows into point uncertainty through the sensor lever arm.
struct PlaneCorrespondenceGate {
    bool enabled=false;
    double squared_sigma=0.0, surface_variance=0.0, angular_variance=0.0;
    Eigen::Matrix3d position_covariance=Eigen::Matrix3d::Zero();
    double SquaredInnovation(const Eigen::Vector3d& predicted_lever,
                             const Eigen::Vector3d& predicted_position,
                             const Eigen::Vector3d& target,
                             const Eigen::Vector3d& normal) const {
        const double residual=normal.dot(predicted_lever+predicted_position-target);
        const double variance=surface_variance+normal.dot(position_covariance*normal)+
            angular_variance*predicted_lever.cross(normal).squaredNorm();
        return variance>0.0?residual*residual/variance:0.0;
    }
    bool Accept(double squared_innovation) const {
        return !enabled || squared_innovation<=squared_sigma;
    }
};
}  // namespace kiss_icp
