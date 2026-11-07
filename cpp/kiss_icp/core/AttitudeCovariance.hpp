// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Eigenvalues>
#include <optional>
namespace kiss_icp {
inline std::optional<Eigen::Matrix3d> ValidAttitudeCovariance(const Eigen::Matrix3d& covariance) {
    if(!covariance.allFinite() || !covariance.isApprox(covariance.transpose(),1e-8))return std::nullopt;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(covariance);
    if(eigen.info()!=Eigen::Success || eigen.eigenvalues().minCoeff()<=0)return std::nullopt;
    return covariance;
}
} // namespace kiss_icp
