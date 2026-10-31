// SPDX-License-Identifier: MIT
#pragma once
#include "kiss_icp/core/InertialPrediction.hpp"
#include <optional>

namespace kiss_icp {
struct ContactSample {
    double time;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    bool valid;
};

// A local displacement, never an absolute leg position. Contact samples may
// arrive after LiDAR registration: update stochastic scan-time clones rather
// than applying an old velocity at the current timestamp.
class ContactDisplacementAid {
public:
    double displacement_sigma=.02;
    double max_joint_gap=.04;
    double max_interval=.25;
    Eigen::Vector3d imu_origin_base=Eigen::Vector3d::Zero();

    void Reset() { last_end_=-1.0;internal_from_leg_.reset(); }
    size_t Update(InertialPrediction& filter,const std::vector<ContactSample>& samples) {
        if(samples.size()<2)return 0;
        const auto& times=filter.clone_times();const auto& rotations=filter.clone_rotations();
        size_t applied=0;
        for(size_t i=1;i<times.size();++i) {
            const double begin=times[i-1],end=times[i];
            if(end<=last_end_)continue;
            if(samples.back().time<end)break;
            // No factor spans unknown support, joint gaps, or an unavailable
            // historical endpoint. The next valid interval starts afresh.
            last_end_=end;
            if(samples.front().time>begin || end-begin>max_interval)continue;
            auto at=[&](double time)->std::optional<ContactSample> {
                auto after=std::lower_bound(samples.begin(),samples.end(),time,
                    [](const ContactSample& s,double t){return s.time<t;});
                if(after==samples.end())return std::nullopt;
                if(after->time==time)return after->valid?std::optional<ContactSample>(*after):std::nullopt;
                if(after==samples.begin())return std::nullopt;
                const auto& before=*std::prev(after);
                const double dt=after->time-before.time;
                if(!before.valid || !after->valid || dt<=0 || dt>max_joint_gap)return std::nullopt;
                const double u=(time-before.time)/dt;
                return ContactSample{time,(1-u)*before.position+u*after->position,
                    before.orientation.slerp(u,after->orientation).normalized(),true};
            };
            const auto first=at(begin),last=at(end);
            if(!first || !last)continue;
            bool supported=true;
            for(size_t j=1;j<samples.size();++j) {
                if(samples[j].time<=begin || samples[j-1].time>=end)continue;
                if(!samples[j].valid || !samples[j-1].valid ||
                   samples[j].time-samples[j-1].time>max_joint_gap) { supported=false;break; }
            }
            if(!supported)continue;
            if(!internal_from_leg_)
                internal_from_leg_=rotations[i-1]*first->orientation.conjugate();
            const Eigen::Vector3d displacement=*internal_from_leg_*(last->position-first->position)+
                rotations[i]*imu_origin_base-rotations[i-1]*imu_origin_base;
            // Only horizontal FK displacement has been calibrated. An
            // uncertain leg height must not become a precise vertical aid.
            const Eigen::Vector3d up=*internal_from_leg_*Eigen::Vector3d::UnitZ();
            const Eigen::Matrix3d horizontal=Eigen::Matrix3d::Identity()-up*up.transpose();
            if(filter.CorrectDisplacement(i-1,i,displacement,
                displacement_sigma*displacement_sigma*Eigen::Matrix3d::Identity(),horizontal))++applied;
        }
        return applied;
    }
private:
    double last_end_=-1.0;
    std::optional<Eigen::Quaterniond> internal_from_leg_;
};
}  // namespace kiss_icp
