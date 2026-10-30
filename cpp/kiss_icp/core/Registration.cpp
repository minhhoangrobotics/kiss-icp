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
#include "Registration.hpp"

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <memory>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>

#include "VoxelHashMap.hpp"
#include "VoxelUtils.hpp"
#include "Surface.hpp"

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

using Correspondences = tbb::concurrent_vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>;
using LinearSystem = std::pair<Eigen::Matrix6d, Eigen::Vector6d>;

namespace {
inline double square(double x) { return x * x; }

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}

Correspondences DataAssociation(const std::vector<Eigen::Vector3d> &points,
                                const kiss_icp::VoxelHashMap &voxel_map,
                                const double max_correspondance_distance) {
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    Correspondences correspondences;
    correspondences.reserve(points.size());
    tbb::parallel_for(
        // Range
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        [&](const tbb::blocked_range<points_iterator> &r) {
            std::for_each(r.begin(), r.end(), [&](const auto &point) {
                const auto &[closest_neighbor, distance] = voxel_map.GetClosestNeighbor(point);
                if (distance < max_correspondance_distance) {
                    correspondences.emplace_back(point, closest_neighbor);
                }
            });
        });
    return correspondences;
}

LinearSystem BuildLinearSystem(const Correspondences &correspondences, const double kernel_scale) {
    auto compute_jacobian_and_residual = [](const auto &correspondence) {
        const auto &[source, target] = correspondence;
        const Eigen::Vector3d residual = source - target;
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source);
        return std::make_tuple(J_r, residual);
    };

    auto sum_linear_systems = [](LinearSystem a, const LinearSystem &b) {
        a.first += b.first;
        a.second += b.second;
        return a;
    };

    auto GM_weight = [&](const double &residual2) {
        return square(kernel_scale) / square(kernel_scale + residual2);
    };

    using correspondence_iterator = Correspondences::const_iterator;
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<correspondence_iterator>{correspondences.cbegin(),
                                                    correspondences.cend()},
        // Identity
        LinearSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero()),
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<correspondence_iterator> &r, LinearSystem J) -> LinearSystem {
            return std::transform_reduce(
                r.begin(), r.end(), J, sum_linear_systems, [&](const auto &correspondence) {
                    const auto &[J_r, residual] = compute_jacobian_and_residual(correspondence);
                    const double w = GM_weight(residual.squaredNorm());
                    return LinearSystem(J_r.transpose() * w * J_r,        // JTJ
                                        J_r.transpose() * w * residual);  // JTr
                });
        },
        // 2nd Lambda: Parallel reduction of the private Jacboians
        sum_linear_systems);

    return {JTJ, JTr};
}

Sophus::SE3d AlignSurfaces(const std::vector<Eigen::Vector3d>& frame,
                           const std::vector<Eigen::Vector3d>& targets,
                           const std::vector<Eigen::Vector3d>& normals,
                           const Sophus::SE3d& initial_guess, double max_distance,
                           double kernel_scale, bool fix_rotation, double rotation_information,
                           int iterations, double convergence, const kiss_icp::SurfaceOptions& options,
                           kiss_icp::SurfaceDiagnostics& diagnostics,
                           const kiss_icp::VoxelHashMap* exact_map = nullptr) {
    std::unique_ptr<kiss_icp::PointIndex> index;
    if (options.covariance_regularization < 1.0) index = std::make_unique<kiss_icp::PointIndex>(targets);
    const auto source_normals = options.covariance_regularization < 1.0 ?
        kiss_icp::EstimateNormals(frame, options.neighbors, options.radius, options.planarity_ratio) :
        std::vector<Eigen::Vector3d>{};
    struct NormalSystem {
        Eigen::Matrix6d H = Eigen::Matrix6d::Zero();
        Eigen::Vector6d b = Eigen::Vector6d::Zero();
        size_t count = 0;
        double error2 = 0.0;
        double point_information = 0.0;
        Eigen::Vector3d point_gradient = Eigen::Vector3d::Zero();
    };
    Sophus::SE3d estimate = initial_guess;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto system = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, frame.size(), 64), NormalSystem{},
            [&](const tbb::blocked_range<size_t>& range, NormalSystem local) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    const auto& point = frame[i];
                    const Eigen::Vector3d lever = estimate.so3() * point;
                    const Eigen::Vector3d source = lever + estimate.translation();
                    Eigen::Vector3d target, normal = Eigen::Vector3d::Zero();
                    if (options.directional_hybrid && exact_map && !options.inertial_translation_valid) {
                        const auto [point_match, point_distance] = exact_map->GetClosestNeighbor(source);
                        if (point_distance < max_distance) {
                            const auto point_residual = (source-point_match).eval();
                            const double point_weight = square(kernel_scale) /
                                square(kernel_scale+point_residual.squaredNorm());
                            local.point_information += point_weight;
                            local.point_gradient += point_weight * point_residual;
                        }
                    }
                    if (options.covariance_regularization >= 1.0 && exact_map) {
                        const auto [matched, distance] = exact_map->GetClosestNeighbor(source, max_distance);
                        if (!(distance < max_distance)) continue;
                        target = matched;
                    } else {
                        const auto near = index->Neighbors(source, 1, max_distance);
                        if (near.empty()) continue;
                        const size_t match = near.front().second;
                        normal = normals[match];
                        target = targets[match];
                    }
                    const Eigen::Vector3d residual = source - target;
                    Eigen::Matrix3d information;
                    if (options.covariance_regularization >= 1.0) {
                        information.setIdentity();
                    } else if (options.covariance_regularization > 0.0) {
                        // GICP: a planar return has small normal variance and large
                        // tangential variance. Combine BOTH measured covariances in
                        // world coordinates. An edge/sparse return uses isotropic
                        // uncertainty, retaining its information instead of dropping it.
                        const double epsilon = options.covariance_regularization;
                        const Eigen::Vector3d source_normal = estimate.so3() * source_normals[i];
                        const Eigen::Matrix3d covariance = 2.0 * Eigen::Matrix3d::Identity() -
                            (1.0-epsilon) * (normal*normal.transpose() +
                                             source_normal*source_normal.transpose());
                        information = 2.0 * epsilon * covariance.inverse();
                    } else {
                        information = normal * normal.transpose();
                    }
                    Eigen::Matrix3_6d J;
                    J.leftCols<3>() = Eigen::Matrix3d::Identity();
                    // Rotate about the sensor, not the arbitrary world origin.
                    J.rightCols<3>() = -Sophus::SO3d::hat(lever);
                    const double residual2 = residual.dot(information * residual);
                    const double weight = square(square(kernel_scale) / (square(kernel_scale) + residual2));
                    local.H.noalias() += weight * J.transpose() * information * J;
                    local.b.noalias() += weight * J.transpose() * information * residual;
                    local.error2 += residual2;
                    ++local.count;
                }
                return local;
            }, [](NormalSystem a, const NormalSystem& other) {
                a.H += other.H;
                a.b += other.b;
                a.count += other.count;
                a.error2 += other.error2;
                a.point_information += other.point_information;
                a.point_gradient += other.point_gradient;
                return a;
            });
        auto& H = system.H;
        auto& b = system.b;
        const auto count = system.count;
        diagnostics.correspondences = count;
        diagnostics.residual_rmse = count ? std::sqrt(system.error2 / static_cast<double>(count)) : 0.0;
        if (count < 6) break;
        if (options.directional_hybrid && (system.point_information > 0.0 || options.inertial_translation_valid)) {
            // Plane normals constrain some translation directions much more
            // strongly than others. Use an independent inertial prior when
            // available, otherwise current-scan point associations, in weak
            // directions. Strong directions and rotation retain the surface
            // metric.
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> geometry(H.topLeftCorner<3,3>());
            if (geometry.info() == Eigen::Success) {
                Eigen::Matrix3d strong = Eigen::Matrix3d::Zero();
                // Compare against the second supported direction. A dense
                // floor must not drown out the information from sparse walls.
                // Require at least one effective unit-normal observation.
                const double threshold = std::max(1.0, options.translation_eigenvalue_ratio *
                                                       geometry.eigenvalues()[1]);
                for (int k=0; k<3; ++k) {
                    if (geometry.eigenvalues()[k] > std::max(1e-9,threshold))
                        strong.noalias() += geometry.eigenvectors().col(k) *
                                            geometry.eigenvectors().col(k).transpose();
                }
                const Eigen::Matrix3d weak = Eigen::Matrix3d::Identity()-strong;
                diagnostics.observable_translation = strong;
                Eigen::Matrix6d projection = Eigen::Matrix6d::Identity();
                projection.topLeftCorner<3,3>() = strong;
                H = (projection * H * projection).eval();
                b = (projection * b).eval();
                if (options.inertial_translation_valid) {
                    // No geometric measurement exists along this subspace.
                    // Keep the independently IMU-propagated translation prior;
                    // discrete floor samples must not invent velocity there.
                    const double information=static_cast<double>(count);
                    H.topLeftCorner<3,3>() += information * weak;
                    b.head<3>() += information * weak * (estimate.translation()-initial_guess.translation());
                } else {
                    H.topLeftCorner<3,3>() += system.point_information * weak;
                    b.head<3>() += weak * system.point_gradient;
                }
            }
        }
        // The absolute scan-time IMU attitude is a measurement, not a clamp.
        H.bottomRightCorner<3, 3>().diagonal().array() += rotation_information;
        b.tail<3>() += rotation_information * (estimate.so3() * initial_guess.so3().inverse()).log();
        if (fix_rotation) {
            H.bottomRows<3>().setZero();
            H.rightCols<3>().setZero();
            b.tail<3>().setZero();
        }
        // Project onto measured directions. Unobservable directions receive
        // zero correction and keep the motion prediction; never invert noise.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(H);
        if (eigen.info() != Eigen::Success) break;
        const double cutoff = std::max(1e-9, options.eigenvalue_ratio * eigen.eigenvalues().maxCoeff());
        Eigen::Vector6d inverse = Eigen::Vector6d::Zero();
        diagnostics.rank = 0;
        for (int k = 0; k < 6; ++k) {
            if (eigen.eigenvalues()[k] > cutoff) {
                inverse[k] = 1.0 / eigen.eigenvalues()[k];
                ++diagnostics.rank;
            }
        }
        const Eigen::Vector6d dx = -eigen.eigenvectors() * inverse.asDiagonal() *
                                   eigen.eigenvectors().transpose() * b;
        if (!dx.allFinite()) break;
        estimate = Sophus::SE3d(Sophus::SO3d::exp(dx.tail<3>()) * estimate.so3(),
                                estimate.translation() + dx.head<3>());
        if (dx.norm() < convergence) break;
    }
    return estimate;
}
}  // namespace

namespace kiss_icp {

Registration::Registration(int max_num_iteration, double convergence_criterion, int max_num_threads)
    : max_num_iterations_(max_num_iteration),
      convergence_criterion_(convergence_criterion),
      // Only manipulate the number of threads if the user specifies something greater than 0
      max_num_threads_(max_num_threads > 0 ? max_num_threads
                                           : tbb::this_task_arena::max_concurrency()) {
    // This global variable requires static duration storage to be able to manipulate the max
    // concurrency from TBB across the entire class
    static const auto tbb_control_settings = tbb::global_control(
        tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_num_threads_));
}

Sophus::SE3d Registration::AlignPointsToMap(const std::vector<Eigen::Vector3d> &frame,
                                            const VoxelHashMap &voxel_map,
                                            const Sophus::SE3d &initial_guess,
                                            const double max_distance,
                                            const double kernel_scale,
                                            bool fix_rotation,
                                            double rotation_prior_information,
                                            const SurfaceOptions& surface) {
    surface_diagnostics = {};
    if (voxel_map.Empty()) return initial_guess;

    if (surface.enabled) {
        if (surface.covariance_regularization >= 1.0) {
            return AlignSurfaces(frame, {}, {}, initial_guess, max_distance, kernel_scale,
                                 fix_rotation, rotation_prior_information, max_num_iterations_,
                                 convergence_criterion_, surface, surface_diagnostics, &voxel_map);
        }
        const auto [targets, normals] = voxel_map.Surfaces(surface.covariance_regularization > 0.0);
        if (targets.size() >= 6) {
            return AlignSurfaces(frame, targets, normals, initial_guess, max_distance,
                                 kernel_scale, fix_rotation, rotation_prior_information,
                                 max_num_iterations_, convergence_criterion_,
                                 surface, surface_diagnostics,
                                 surface.directional_hybrid ? &voxel_map : nullptr);
        }
        // A genuinely non-planar scene still supports the original KISS metric.
        // Rank-deficient planar scenes do not take this fallback.
        surface_diagnostics.fallback = true;
    }

    // Equation (9)
    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    // ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < max_num_iterations_; ++j) {
        // Equation (10)
        const auto correspondences = DataAssociation(source, voxel_map, max_distance);
        if (correspondences.size() < 6) break;
        // Equation (11)
        auto [JTJ, JTr] = BuildLinearSystem(correspondences, kernel_scale);
        if (rotation_prior_information > 0.0) {
            // Use the scan-time IMU as an orientation measurement. Penalize
            // deviation from it while allowing geometry to correct the pose.
            // ICP uses left increments, so the prior residual is expressed in
            // the same world tangent as the rotational Jacobian.
            const Eigen::Vector3d rotation_error =
                (T_icp.so3()).log();
            JTJ.bottomRightCorner<3, 3>().diagonal().array() += rotation_prior_information;
            JTr.tail<3>() += rotation_prior_information * rotation_error;
        }
        Eigen::Vector6d dx = Eigen::Vector6d::Zero();
        if (fix_rotation) {
            // The scan-time IMU already supplies orientation. Solving all six
            // axes here lets small yaw corrections accumulate during turns,
            // even when every scan has good overlap with the *drifting* local map.
            // Solve translation only, with the same robust point weights.
            dx.head<3>() = JTJ.topLeftCorner<3, 3>().ldlt().solve(-JTr.head<3>());
        } else {
            dx = JTJ.ldlt().solve(-JTr);
        }
        if (!dx.allFinite()) break;
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        // Equation (12)
        TransformPoints(estimation, source);
        // Update iterations
        T_icp = estimation * T_icp;
        // Termination criteria
        if (dx.norm() < convergence_criterion_) break;
    }
    // Spit the final transformation
    return T_icp * initial_guess;
}

}  // namespace kiss_icp
