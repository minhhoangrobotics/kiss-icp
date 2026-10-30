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

#include "KissICP.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"
#include "kiss_icp/core/Surface.hpp"

namespace kiss_icp::pipeline {

KissICP::Vector3dVectorTuple KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                                    const std::vector<double> &timestamps,
                                                    const std::optional<Sophus::SE3d> &pose_hint,
                                                    bool imu_orientation_valid,
                                                    bool inertial_translation_valid,
                                                    const std::function<Sophus::SE3d(const Sophus::SE3d&,const SurfaceDiagnostics&)>& state_update,
                                                    const std::optional<Eigen::Matrix3d>& inertial_covariance) {
    // Preprocess the input cloud
    const auto &preprocessed_frame = preprocessor_.Preprocess(frame, timestamps, last_delta_);

    // Voxelize
    const auto &[source, frame_downsample] = Voxelize(preprocessed_frame);

    // Get adaptive_threshold
    const double sigma = adaptive_threshold_.ComputeThreshold();

    // Compute initial_guess for ICP
    const auto initial_guess = pose_hint.value_or(last_pose_ * last_delta_);

    // Run ICP
    const bool imu_prior_valid = imu_orientation_valid && pose_hint.has_value();
    auto surface = config_.surface;
    surface.inertial_translation_valid = inertial_translation_valid && imu_prior_valid;
    if(inertial_covariance && surface.inertial_translation_valid) {
        surface.inertial_information=config_.voxel_size*config_.voxel_size*
            inertial_covariance->ldlt().solve(Eigen::Matrix3d::Identity());
    }
    auto new_pose = registration_.AlignPointsToMap(
        source, local_map_, initial_guess, 3.0 * sigma, sigma,
        config_.fix_rotation_from_hint && imu_prior_valid,
        imu_prior_valid ? config_.rotation_prior_information : 0.0, surface);
    registration_.surface_diagnostics.translation_noise_variance=config_.voxel_size*config_.voxel_size;

    last_quality_ = RegistrationQuality{};
    if (config_.surface.enabled && !local_map_.Empty() &&
        !registration_.surface_diagnostics.fallback &&
        registration_.surface_diagnostics.correspondences < 6) {
        last_quality_.accepted = false;
        last_quality_.rejection_reason = "insufficient surface correspondences";
        return {preprocessed_frame, source};
    }
    auto validate_pose=[&](const Sophus::SE3d& candidate) {
        last_quality_=RegistrationQuality{};
        if(!candidate.matrix().allFinite()) {
            last_quality_.accepted=false;last_quality_.inlier_ratio=0;
            last_quality_.rmse=std::numeric_limits<double>::infinity();
            last_quality_.rejection_reason="non-finite accepted pose";return false;
        }
        if(!config_.quality_gate || local_map_.Empty())return true;
        const auto difference = initial_guess.inverse() * candidate;
        last_quality_.translation_deviation = difference.translation().norm();
        last_quality_.rotation_deviation = difference.so3().log().norm();
        last_quality_.vertical_deviation =
            std::abs(candidate.translation().z() - initial_guess.translation().z());
        size_t sampled = 0, inliers = 0;
        double error2 = 0.0;
        const size_t stride = std::max<size_t>(1, source.size() / 300);
        for (size_t i = 0; i < source.size(); i += stride) {
            ++sampled;
            const auto &point = source[i];
            if (!point.allFinite()) continue;
            const auto &[neighbor, distance] = local_map_.GetClosestNeighbor(candidate * point);
            (void)neighbor;
            if (distance < config_.inlier_distance) {
                ++inliers;
                error2 += distance * distance;
            }
        }
        last_quality_.inlier_ratio = sampled ? static_cast<double>(inliers) /
                                               static_cast<double>(sampled) : 0.0;
        last_quality_.rmse = inliers ? std::sqrt(error2 / static_cast<double>(inliers)) :
                                      std::numeric_limits<double>::infinity();
        if (last_quality_.inlier_ratio < config_.min_inlier_ratio) {
            last_quality_.rejection_reason = "low local-map inlier ratio";
        } else if (last_quality_.rmse > config_.max_inlier_rmse) {
            last_quality_.rejection_reason = "high local-map RMSE";
        } else if (pose_hint && last_quality_.translation_deviation > config_.max_prior_translation) {
            last_quality_.rejection_reason = "translation disagrees with motion prediction";
        } else if (pose_hint && last_quality_.rotation_deviation > config_.max_prior_rotation) {
            last_quality_.rejection_reason = "rotation disagrees with IMU";
        }
        last_quality_.accepted = last_quality_.rejection_reason.empty();
        return last_quality_.accepted;
    };
    // Validate the actual LiDAR measurement before it can update velocity or
    // bias. A small Kalman gain must not conceal an invalid ICP measurement.
    if(!validate_pose(new_pose))return {preprocessed_frame,source};
    if(state_update) {
        new_pose=state_update(new_pose,registration_.surface_diagnostics);
        if(!validate_pose(new_pose))return {preprocessed_frame,source};
    }

    // Compute the difference between the prediction and the actual estimate
    const auto model_deviation = initial_guess.inverse() * new_pose;

    // Update step: threshold, local map, delta, and the last pose
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    if (config_.surface.enabled && config_.surface.covariance_regularization < 1.0) {
        const auto normals = EstimateNormals(frame_downsample, config_.surface.neighbors,
                                             config_.surface.radius, config_.surface.planarity_ratio);
        local_map_.Update(frame_downsample, new_pose, normals);
    } else {
        local_map_.Update(frame_downsample, new_pose);
    }
    last_delta_ = last_pose_.inverse() * new_pose;
    last_pose_ = new_pose;

    // Return the (deskew) input raw scan (preprocessed_frame) and the points used for registration
    // (source)
    return {preprocessed_frame, source};
}

KissICP::Vector3dVectorTuple KissICP::Voxelize(const std::vector<Eigen::Vector3d> &frame) const {
    const auto voxel_size = config_.voxel_size;
    const auto frame_downsample = kiss_icp::VoxelDownsample(frame, voxel_size * 0.5);
    const auto source = kiss_icp::VoxelDownsample(frame_downsample, voxel_size * 1.5);
    return {source, frame_downsample};
}
void KissICP::Reset() {
    last_pose_ = Sophus::SE3d();
    last_delta_ = Sophus::SE3d();

    // Clear the local map
    local_map_.Clear();

    // Reset adaptive threshold (it will start fresh)
    adaptive_threshold_ =
        AdaptiveThreshold(config_.initial_threshold, config_.min_motion_th, config_.max_range);
    last_quality_ = RegistrationQuality{};
}

void KissICP::ResetToPose(const Sophus::SE3d &pose) {
    Reset();
    last_pose_ = pose;
}

}  // namespace kiss_icp::pipeline
