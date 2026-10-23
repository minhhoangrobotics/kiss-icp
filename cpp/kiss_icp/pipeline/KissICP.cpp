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

namespace kiss_icp::pipeline {

KissICP::Vector3dVectorTuple KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                                    const std::vector<double> &timestamps,
                                                    const std::optional<Sophus::SE3d> &pose_hint) {
    // Preprocess the input cloud
    const auto &preprocessed_frame = preprocessor_.Preprocess(frame, timestamps, last_delta_);

    // Voxelize
    const auto &[source, frame_downsample] = Voxelize(preprocessed_frame);

    // Get adaptive_threshold
    const double sigma = adaptive_threshold_.ComputeThreshold();

    // Compute initial_guess for ICP
    const auto initial_guess = pose_hint.value_or(last_pose_ * last_delta_);

    // Run ICP
    const auto new_pose = registration_.AlignPointsToMap(source,         // frame
                                                         local_map_,     // voxel_map
                                                         initial_guess,  // initial_guess
                                                         3.0 * sigma,    // max_correspondence_dist
                                                         sigma);         // kernel

    last_quality_ = RegistrationQuality{};
    if (config_.quality_gate && !local_map_.Empty()) {
        if (!new_pose.matrix().allFinite()) {
            last_quality_.accepted = false;
            last_quality_.inlier_ratio = 0.0;
            last_quality_.rmse = std::numeric_limits<double>::infinity();
            return {preprocessed_frame, source};
        }
        const auto difference = initial_guess.inverse() * new_pose;
        last_quality_.translation_deviation = difference.translation().norm();
        last_quality_.rotation_deviation = difference.so3().log().norm();
        size_t sampled = 0, inliers = 0;
        double error2 = 0.0;
        const size_t stride = std::max<size_t>(1, source.size() / 300);
        for (size_t i = 0; i < source.size(); i += stride) {
            ++sampled;
            const auto &point = source[i];
            if (!point.allFinite()) continue;
            const auto &[neighbor, distance] = local_map_.GetClosestNeighbor(new_pose * point);
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
        last_quality_.accepted = last_quality_.inlier_ratio >= config_.min_inlier_ratio &&
            last_quality_.rmse <= config_.max_inlier_rmse &&
            (!pose_hint || (last_quality_.translation_deviation <= config_.max_prior_translation &&
                            last_quality_.rotation_deviation <= config_.max_prior_rotation));
        if (!last_quality_.accepted) return {preprocessed_frame, source};
    }

    // Compute the difference between the prediction and the actual estimate
    const auto model_deviation = initial_guess.inverse() * new_pose;

    // Update step: threshold, local map, delta, and the last pose
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    local_map_.Update(frame_downsample, new_pose);
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

}  // namespace kiss_icp::pipeline
