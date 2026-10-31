// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <vector>

namespace kiss_icp {
struct InertialSample {
    double time;
    Eigen::Quaterniond orientation;  // odometry <- IMU, from the attitude estimator
    Eigen::Vector3d acceleration;   // specific force, IMU coordinates, m/s^2
};

// Translational error-state filter. Attitude is supplied by the IMU EKF;
// state is IMU position, world velocity, and accelerometer bias in IMU axes.
// LiDAR updates only the directions actually constrained by surface geometry.
class InertialPrediction {
public:
    double acceleration_noise = 0.2;
    double bias_walk = 0.005;
    double position_noise = 0.02;
    double max_sample_gap = 0.05;

    void Reset() { initialized_ = false; }
    void Initialize(double time, const Eigen::Vector3d& position,
                    const Eigen::Vector3d& gravity) {
        time_ = time;
        gravity_ = gravity;
        state_.setZero();
        state_.head<3>() = position;
        covariance_.setZero();
        covariance_.diagonal().head<3>().setConstant(position_noise*position_noise);
        covariance_.diagonal().segment<3>(3).setConstant(1.0);
        covariance_.diagonal().tail<3>().setConstant(0.01);
        initialized_ = true;
        clone_times_.clear();clone_rotations_.clear();
        clone_state_.resize(0);clone_covariance_.resize(0,0);state_clone_covariance_.resize(9,0);
    }
    bool initialized() const { return initialized_; }
    const Eigen::Vector3d position() const { return state_.head<3>(); }
    const Eigen::Vector3d velocity() const { return state_.segment<3>(3); }
    const Eigen::Vector3d bias() const { return state_.tail<3>(); }
    const Eigen::Matrix<double,9,9>& covariance() const { return covariance_; }
    const std::vector<double>& clone_times() const { return clone_times_; }
    const std::vector<Eigen::Quaterniond>& clone_rotations() const { return clone_rotations_; }
    Eigen::VectorXd joint_state() const {
        Eigen::VectorXd x(9+clone_state_.size());x.head<9>()=state_;x.tail(clone_state_.size())=clone_state_;return x;
    }
    Eigen::MatrixXd joint_covariance() const {
        const Eigen::Index n=clone_state_.size();Eigen::MatrixXd P(9+n,9+n);
        P.topLeftCorner<9,9>()=covariance_;P.topRightCorner(9,n)=state_clone_covariance_;
        P.bottomLeftCorner(n,9)=state_clone_covariance_.transpose();P.bottomRightCorner(n,n)=clone_covariance_;return P;
    }

    // Stochastic position clones retain the cross-covariance needed for delayed
    // contact displacement factors. Shared endpoints are not independent poses.
    void Clone(double time,const Eigen::Quaterniond& base_rotation,size_t limit=8) {
        if(!initialized_ || std::abs(time-time_)>1e-6 ||
           (!clone_times_.empty() && time<=clone_times_.back()))return;
        const Eigen::Index n=clone_state_.size();
        Eigen::MatrixXd next=Eigen::MatrixXd::Zero(n+3,n+3);
        next.topLeftCorner(n,n)=clone_covariance_;
        next.topRightCorner(n,3)=state_clone_covariance_.topRows<3>().transpose();
        next.bottomLeftCorner(3,n)=state_clone_covariance_.topRows<3>();
        next.bottomRightCorner<3,3>()=covariance_.topLeftCorner<3,3>();
        clone_covariance_=std::move(next);
        clone_state_.conservativeResize(n+3);clone_state_.tail<3>()=position();
        state_clone_covariance_.conservativeResize(9,n+3);
        state_clone_covariance_.rightCols<3>()=covariance_.leftCols<3>();
        clone_times_.push_back(time);clone_rotations_.push_back(base_rotation.normalized());
        while(clone_times_.size()>limit) {
            const Eigen::Index remaining=clone_state_.size()-3;
            clone_state_=clone_state_.tail(remaining).eval();
            clone_covariance_=clone_covariance_.bottomRightCorner(remaining,remaining).eval();
            state_clone_covariance_=state_clone_covariance_.rightCols(remaining).eval();
            clone_times_.erase(clone_times_.begin());clone_rotations_.erase(clone_rotations_.begin());
        }
    }

    bool CorrectDisplacement(size_t from,size_t to,const Eigen::Vector3d& displacement,
                             const Eigen::Matrix3d& noise,
                             const Eigen::Matrix3d& observable=Eigen::Matrix3d::Identity()) {
        if(from>=to || to>=clone_times_.size() || !displacement.allFinite() ||
           !noise.allFinite() || !observable.allFinite() || !noise.isApprox(noise.transpose()) || noise.selfadjointView<Eigen::Lower>().eigenvalues().minCoeff()<=0)return false;
        Eigen::MatrixXd H=Eigen::MatrixXd::Zero(3,clone_state_.size());
        H.block<3,3>(0,static_cast<Eigen::Index>(3*from))=-observable;
        H.block<3,3>(0,static_cast<Eigen::Index>(3*to))=observable;
        const Eigen::Vector3d residual=observable*displacement-H*clone_state_;
        const Eigen::Matrix3d S=H*clone_covariance_*H.transpose()+noise;
        const Eigen::LDLT<Eigen::Matrix3d> ldlt(S);
        if(ldlt.info()!=Eigen::Success || !ldlt.isPositive())return false;
        const Eigen::Matrix3d inverse=ldlt.solve(Eigen::Matrix3d::Identity());
        const Eigen::Matrix<double,9,3> K=state_clone_covariance_*H.transpose()*inverse;
        const Eigen::MatrixXd past_K=clone_covariance_*H.transpose()*inverse;
        const auto next_state=(state_+K*residual).eval();
        const auto next_clone=(clone_state_+past_K*residual).eval();
        const auto next_cross=(state_clone_covariance_-K*H*clone_covariance_).eval();
        const auto next_covariance=(covariance_-K*S*K.transpose()).eval();
        const auto next_clone_covariance=(clone_covariance_-past_K*S*past_K.transpose()).eval();
        if(!next_state.allFinite()||!next_clone.allFinite()||!next_cross.allFinite()||
           !next_covariance.allFinite()||!next_clone_covariance.allFinite())return false;
        state_=next_state;clone_state_=next_clone;state_clone_covariance_=next_cross;
        covariance_=.5*(next_covariance+next_covariance.transpose());
        clone_covariance_=.5*(next_clone_covariance+next_clone_covariance.transpose());
        return true;
    }

    bool Predict(double target, const std::vector<InertialSample>& samples) {
        if (!initialized_ || target < time_ || samples.size() < 2) return false;
        if (target == time_) return true;
        if (samples.front().time > time_ || samples.back().time < target) return false;
        auto upper = std::upper_bound(samples.begin(), samples.end(), time_,
            [](double time,const InertialSample& sample){return time < sample.time;});
        if (upper == samples.begin()) return false;
        size_t i = static_cast<size_t>(upper-samples.begin()-1);
        // Validate the entire interval before changing state: no silent
        // extrapolation or partial advance across missing IMU measurements.
        for (size_t j=i; j+1<samples.size() && samples[j].time<target; ++j) {
            const double dt=samples[j+1].time-samples[j].time;
            if (!(dt>0.0) || dt>max_sample_gap) return false;
        }
        while (time_ < target && i+1 < samples.size()) {
            const auto& a=samples[i]; const auto& b=samples[i+1];
            const double end=std::min(target,b.time);
            const double dt=end-time_;
            const double u=(0.5*(time_+end)-a.time)/(b.time-a.time);
            const Eigen::Matrix3d R=a.orientation.slerp(u,b.orientation).normalized().toRotationMatrix();
            const Eigen::Vector3d specific=(1.0-u)*a.acceleration+u*b.acceleration;
            const Eigen::Vector3d acceleration=R*(specific-state_.tail<3>())+gravity_;
            state_.head<3>() += state_.segment<3>(3)*dt+0.5*acceleration*dt*dt;
            state_.segment<3>(3) += acceleration*dt;
            Eigen::Matrix<double,9,9> F=Eigen::Matrix<double,9,9>::Identity();
            F.block<3,3>(0,3)=dt*Eigen::Matrix3d::Identity();
            F.block<3,3>(0,6)=-0.5*dt*dt*R;
            F.block<3,3>(3,6)=-dt*R;
            covariance_=(F*covariance_*F.transpose()).eval();
            state_clone_covariance_=(F*state_clone_covariance_).eval();
            const double variance=acceleration_noise*acceleration_noise;
            covariance_.block<3,3>(0,0).diagonal().array()+=variance*dt*dt*dt/3.0;
            covariance_.block<3,3>(0,3).diagonal().array()+=variance*dt*dt/2.0;
            covariance_.block<3,3>(3,0).diagonal().array()+=variance*dt*dt/2.0;
            covariance_.block<3,3>(3,3).diagonal().array()+=variance*dt;
            covariance_.block<3,3>(6,6).diagonal().array()+=bias_walk*bias_walk*dt;
            time_=end;
            ++i;
        }
        return time_ == target && state_.allFinite() && covariance_.allFinite();
    }

    void Correct(const Eigen::Vector3d& imu_position, const Eigen::Matrix3d& observable) {
        Eigen::Matrix<double,3,9> H=Eigen::Matrix<double,3,9>::Zero();
        H.leftCols<3>()=observable;
        const Eigen::Matrix3d R=position_noise*position_noise*Eigen::Matrix3d::Identity();
        const Eigen::Matrix3d S=H*covariance_*H.transpose()+R;
        const Eigen::Matrix<double,9,3> K=covariance_*H.transpose()*S.ldlt().solve(Eigen::Matrix3d::Identity());
        const Eigen::Vector3d residual=observable*(imu_position-state_.head<3>());
        UpdateClones(H,residual,S,K);
        state_+=K*residual;
        const Eigen::Matrix<double,9,9> A=Eigen::Matrix<double,9,9>::Identity()-K*H;
        covariance_=(A*covariance_*A.transpose()+K*R*K.transpose()).eval();
        covariance_=(0.5*(covariance_+covariance_.transpose())).eval();
    }

    // Plane likelihood in independent eigen-directions. Sparse but nonzero
    // geometric information receives larger noise, not a binary rejection.
    void CorrectInformation(const Eigen::Vector3d& imu_position,
                            const Eigen::Matrix3d& information) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(information);
        if(eigen.info()!=Eigen::Success)return;
        Eigen::Matrix<double,3,9> H=Eigen::Matrix<double,3,9>::Zero();
        Eigen::Matrix3d R=Eigen::Matrix3d::Identity();
        for(int k=0;k<3;++k)if(eigen.eigenvalues()[k]>1e-9) {
            H.block<1,3>(k,0)=eigen.eigenvectors().col(k).transpose();
            R(k,k)=position_noise*position_noise+1.0/eigen.eigenvalues()[k];
        }
        const Eigen::Matrix3d S=H*covariance_*H.transpose()+R;
        const Eigen::Matrix<double,9,3> K=covariance_*H.transpose()*S.ldlt().solve(Eigen::Matrix3d::Identity());
        const Eigen::Vector3d residual=H.leftCols<3>()*(imu_position-state_.head<3>());
        UpdateClones(H,residual,S,K);
        state_+=K*residual;
        const Eigen::Matrix<double,9,9> A=Eigen::Matrix<double,9,9>::Identity()-K*H;
        covariance_=(A*covariance_*A.transpose()+K*R*K.transpose()).eval();
        covariance_=(0.5*(covariance_+covariance_.transpose())).eval();
    }
private:
    void UpdateClones(const Eigen::Matrix<double,3,9>& H,const Eigen::Vector3d& residual,
                      const Eigen::Matrix3d& S,const Eigen::Matrix<double,9,3>& K) {
        if(clone_state_.size()==0)return;
        const Eigen::MatrixXd past_K=state_clone_covariance_.transpose()*H.transpose()*
            S.ldlt().solve(Eigen::Matrix3d::Identity());
        clone_state_+=past_K*residual;
        clone_covariance_=(clone_covariance_-past_K*S*past_K.transpose()).eval();
        clone_covariance_=(.5*(clone_covariance_+clone_covariance_.transpose())).eval();
        state_clone_covariance_=(state_clone_covariance_-K*H*state_clone_covariance_).eval();
    }
    std::vector<double> clone_times_;
    std::vector<Eigen::Quaterniond> clone_rotations_;
    Eigen::VectorXd clone_state_;
    Eigen::MatrixXd clone_covariance_;
    Eigen::MatrixXd state_clone_covariance_=Eigen::MatrixXd(9,0);
    bool initialized_=false;
    double time_=0.0;
    Eigen::Vector3d gravity_=Eigen::Vector3d(0,0,-9.806);
    Eigen::Matrix<double,9,1> state_=Eigen::Matrix<double,9,1>::Zero();
    Eigen::Matrix<double,9,9> covariance_=Eigen::Matrix<double,9,9>::Identity();
};
}  // namespace kiss_icp
