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
#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

#include "VoxelHashMap.hpp"

namespace kiss_icp {

struct SurfaceOptions {
    bool enabled = false;
    int neighbors = 10;
    double radius = 1.2;
    double planarity_ratio = 0.15;
    double eigenvalue_ratio = 1e-4;
    double covariance_regularization = 0.01;
    bool directional_hybrid = false;
    double translation_eigenvalue_ratio = 0.02;
    bool inertial_translation_valid = false;
};

struct SurfaceDiagnostics {
    size_t correspondences = 0;
    int rank = 0;
    bool fallback = false;
    double residual_rmse = 0.0;
    Eigen::Matrix3d observable_translation = Eigen::Matrix3d::Identity();
};

struct Registration {
    explicit Registration(int max_num_iteration, double convergence_criterion, int max_num_threads);

    Sophus::SE3d AlignPointsToMap(const std::vector<Eigen::Vector3d> &frame,
                                  const VoxelHashMap &voxel_map,
                                  const Sophus::SE3d &initial_guess,
                                  const double max_correspondence_distance,
                                  const double kernel_scale,
                                  bool fix_rotation = false,
                                  double rotation_prior_information = 0.0,
                                  const SurfaceOptions &surface = {});

    SurfaceDiagnostics surface_diagnostics;

    int max_num_iterations_;
    double convergence_criterion_;
    int max_num_threads_;
};
}  // namespace kiss_icp
