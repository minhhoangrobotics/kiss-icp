// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>
#include <tbb/parallel_invoke.h>

namespace kiss_icp {

// Balanced index over immutable points. Owned by a single registration call;
// queries are const, so correspondence searches can run concurrently.
class PointIndex {
public:
    explicit PointIndex(const std::vector<Eigen::Vector3d>& points) : points_(points) {
        order_.resize(points.size());
        std::iota(order_.begin(), order_.end(), size_t{0});
        nodes_.resize(points.size());
        root_ = Build(0, order_.size(), 0);
    }

    std::vector<std::pair<double, size_t>> Neighbors(const Eigen::Vector3d& query,
                                                     size_t count, double radius) const {
        Heap heap;
        if (count > 0) Search(root_, query, count, radius * radius, heap);
        std::vector<std::pair<double, size_t>> result(heap.size());
        for (size_t i = result.size(); i > 0; --i) {
            result[i - 1] = heap.top();
            heap.pop();
        }
        return result;
    }

private:
    struct Node { size_t point; int left; int right; int axis; };
    using Heap = std::priority_queue<std::pair<double, size_t>>;
    int Build(size_t begin, size_t end, int depth) {
        if (begin == end) return -1;
        const size_t mid = begin + (end - begin) / 2;
        const int axis = depth % 3;
        std::nth_element(order_.begin() + static_cast<std::ptrdiff_t>(begin),
                         order_.begin() + static_cast<std::ptrdiff_t>(mid),
                         order_.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](size_t a, size_t b) { return points_[a][axis] < points_[b][axis]; });
        const int node = static_cast<int>(mid);
        nodes_[mid] = {order_[mid], -1, -1, axis};
        int left = -1, right = -1;
        // Child ranges and node slots are disjoint. Parallel construction
        // preserves the same partition/tree while honoring KISS's TBB limit.
        if (depth < 2 && end-begin > 10000) {
            tbb::parallel_invoke([&] { left = Build(begin, mid, depth+1); },
                                 [&] { right = Build(mid+1, end, depth+1); });
        } else {
            left = Build(begin, mid, depth + 1);
            right = Build(mid + 1, end, depth + 1);
        }
        nodes_[static_cast<size_t>(node)].left = left;
        nodes_[static_cast<size_t>(node)].right = right;
        return node;
    }
    void Search(int node, const Eigen::Vector3d& query, size_t count,
                double radius2, Heap& heap) const {
        if (node < 0) return;
        const auto& n = nodes_[static_cast<size_t>(node)];
        const auto delta = (query - points_[n.point]).eval();
        const double distance2 = delta.squaredNorm();
        if (distance2 <= radius2) {
            if (heap.size() < count) heap.emplace(distance2, n.point);
            else if (distance2 < heap.top().first) {
                heap.pop();
                heap.emplace(distance2, n.point);
            }
        }
        const double split = delta[n.axis];
        Search(split < 0.0 ? n.left : n.right, query, count, radius2, heap);
        const double limit = heap.size() == count ? std::min(radius2, heap.top().first) : radius2;
        if (split * split <= limit)
            Search(split < 0.0 ? n.right : n.left, query, count, radius2, heap);
    }
    const std::vector<Eigen::Vector3d>& points_;
    std::vector<size_t> order_;
    std::vector<Node> nodes_;
    int root_ = -1;
};

// A zero normal marks an edge, sparse neighborhood, or non-planar return.
// Estimate on one scan, never on an already blurred accumulated map.
inline std::vector<Eigen::Vector3d> EstimateNormals(
    const std::vector<Eigen::Vector3d>& points, int neighbors, double radius,
    double planarity_ratio) {
    PointIndex index(points);
    std::vector<Eigen::Vector3d> normals(points.size(), Eigen::Vector3d::Zero());
    for (size_t i = 0; i < points.size(); ++i) {
        const auto near = index.Neighbors(points[i], static_cast<size_t>(neighbors), radius);
        if (near.size() < static_cast<size_t>(neighbors)) continue;
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        for (const auto& item : near) mean += points[item.second];
        mean /= static_cast<double>(near.size());
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (const auto& item : near) {
            const auto d = (points[item.second] - mean).eval();
            covariance.noalias() += d * d.transpose();
        }
        covariance /= static_cast<double>(near.size());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(covariance);
        if (eigen.info() != Eigen::Success) continue;
        const auto& values = eigen.eigenvalues();
        // Reject collinear points as well as corners. The second eigenvalue
        // must span at least a few centimetres to define a plane normal.
        if (values[1] > 0.001 && values[0] < planarity_ratio * values[1])
            normals[i] = eigen.eigenvectors().col(0);
    }
    return normals;
}
}  // namespace kiss_icp
