// SPDX-License-Identifier: MIT
#pragma once
#include "kiss_icp/core/Registration.hpp"
#include <Eigen/Eigenvalues>
#include <optional>

namespace kiss_icp {
struct LidarPoseLikelihood {
    Sophus::SE3d pose;
    // ROS fixed-axis covariance [position in world, rotation in world].
    Eigen::Matrix<double,6,6> covariance;
};
// No inertial, contact, or numerical tangent prior is exported as LiDAR data.
inline std::optional<LidarPoseLikelihood> IndependentLidarLikelihood(
    const SurfaceDiagnostics& geometry,const Sophus::SE3d& base_from_lidar,
    const Sophus::SE3d& output_from_internal) {
    using Matrix6=Eigen::Matrix<double,6,6>;
    if(geometry.fallback || geometry.correspondences<6 ||
       !geometry.pose_information.allFinite() || !geometry.pose_gradient.allFinite() ||
       !(geometry.translation_noise_variance>0.0))return std::nullopt;
    Eigen::SelfAdjointEigenSolver<Matrix6> eigen(geometry.pose_information);
    if(eigen.info()!=Eigen::Success || eigen.eigenvalues().maxCoeff()<=1e-9)return std::nullopt;
    Eigen::Matrix<double,6,1> inverse=Eigen::Matrix<double,6,1>::Zero();
    Eigen::Matrix<double,6,1> variance=Eigen::Matrix<double,6,1>::Constant(1e6);
    const double cutoff=std::max(1e-9,1e-9*eigen.eigenvalues().maxCoeff());
    for(int k=0;k<6;++k)if(eigen.eigenvalues()[k]>cutoff) {
        inverse[k]=1.0/eigen.eigenvalues()[k];
        variance[k]=geometry.translation_noise_variance*inverse[k];
    }
    const Eigen::Matrix<double,6,1> delta=-eigen.eigenvectors()*inverse.asDiagonal()*
        eigen.eigenvectors().transpose()*geometry.pose_gradient;
    if(!delta.allFinite())return std::nullopt;
    const Sophus::SE3d lidar(Sophus::SO3d::exp(delta.tail<3>())*geometry.linearization_rotation,
                             geometry.linearization_position+delta.head<3>());
    const Sophus::SE3d base=lidar*base_from_lidar.inverse();
    Matrix6 J=Matrix6::Identity();
    J.topRightCorner<3,3>()=Sophus::SO3d::hat(base.so3()*base_from_lidar.translation());
    Matrix6 Q=Matrix6::Zero();
    Q.topLeftCorner<3,3>()=output_from_internal.rotationMatrix();
    Q.bottomRightCorner<3,3>()=output_from_internal.rotationMatrix();
    const Matrix6 map=Q*J*eigen.eigenvectors();
    Matrix6 covariance=map*variance.asDiagonal()*map.transpose();
    covariance=(.5*(covariance+covariance.transpose())).eval();
    return LidarPoseLikelihood{output_from_internal*base,covariance};
}
} // namespace kiss_icp
