// SPDX-License-Identifier: MIT
#pragma once
#include "kiss_icp/core/InertialPrediction.hpp"
#include "kiss_icp/core/Registration.hpp"
#include <sophus/se3.hpp>
namespace kiss_icp {
// Stage this update on a copy; commit only after the returned pose passes the
// registration gate. Both map insertion and published odometry use this state.
inline Sophus::SE3d FuseInertialLidarPose(InertialPrediction& posterior,
                                        const Sophus::SE3d& measured_lidar,
                                        const Sophus::SE3d& base_from_lidar,
                                        const Eigen::Vector3d& imu_origin_base,
                                        const Eigen::Matrix3d& observable) {
    Sophus::SE3d base=measured_lidar*base_from_lidar.inverse();
    posterior.Correct(base.translation()+base.so3()*imu_origin_base,observable);
    base.translation()=posterior.position()-base.so3()*imu_origin_base;
    return base*base_from_lidar;
}

inline Sophus::SE3d FuseInertialLidarLikelihood(InertialPrediction& posterior,
                                              const Sophus::SE3d& candidate,
                                              const Sophus::SE3d& base_from_lidar,
                                              const Eigen::Vector3d& imu_origin_base,
                                              const SurfaceDiagnostics& geometry) {
    if(!geometry.has_translation_measurement)
        return FuseInertialLidarPose(posterior,candidate,base_from_lidar,imu_origin_base,
                                     geometry.observable_translation);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(geometry.translation_information);
    if(eigen.info()!=Eigen::Success)return candidate;
    Eigen::Vector3d inverse=Eigen::Vector3d::Zero();
    for(int k=0;k<3;++k)if(eigen.eigenvalues()[k]>1e-9)inverse[k]=1.0/eigen.eigenvalues()[k];
    const Eigen::Vector3d rotation_delta=(candidate.so3()*geometry.linearization_rotation.inverse()).log();
    // Conditional LiDAR measurement at the optimized attitude, with the
    // inertial prior removed. Null directions are ignored by CorrectInformation.
    const Eigen::Vector3d lidar_measurement=geometry.linearization_position-
        eigen.eigenvectors()*inverse.asDiagonal()*eigen.eigenvectors().transpose()*
        (geometry.translation_gradient+geometry.translation_rotation_information*rotation_delta);
    Sophus::SE3d base=candidate*base_from_lidar.inverse();
    const Eigen::Vector3d measurement=lidar_measurement+base.so3()*
        (imu_origin_base-base_from_lidar.translation());
    posterior.CorrectInformation(measurement,
        geometry.translation_information/geometry.translation_noise_variance);
    base.translation()=posterior.position()-base.so3()*imu_origin_base;
    return base*base_from_lidar;
}
}  // namespace kiss_icp
