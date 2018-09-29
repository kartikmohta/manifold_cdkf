#pragma once

#include "element_types/quaternion.hpp"

template <typename State, typename Input, typename ProcessNoiseVec>
State ProcessModelIMU(const State &state, const Input &u,
                      const ProcessNoiseVec &w, double dt)
{
  using Scalar = typename State::Scalar;
  using Vec3 = typename State::template Vec<3>;

  Vec3 const accel_body =
      u.template topRows<3>() - state.getBiasAccel() + w.template segment<3>(0);
  Vec3 const ang_vel = u.template bottomRows<3>() - state.getBiasGyro() +
                       w.template segment<3>(3);

  Vec3 const accel_grav(0, 0, -9.81);
  Vec3 const accel_world = state.getOrientation() * accel_body + accel_grav;

  Scalar const d_theta = ang_vel.norm() * dt;
  Vec3 const ang_vel_dir =
      (d_theta > 1e-6) ? ang_vel.normalized() : Vec3::Zero();
  Vec3 const ang_vel_vec =
      QuaternionElement<Scalar>::angle_axis_to_vec(d_theta, ang_vel_dir);

  typename State::TangentVec dx = State::TangentVec::Zero();

  dx.segment<3>(0) = state.getVelocity() * dt + 0.5 * accel_world * dt * dt;
  dx.segment<3>(3) = accel_world * dt;
  dx.segment<3>(6) = ang_vel_vec;
  dx.segment<3>(9) = w.segment<3>(6) * dt;
  dx.segment<3>(12) = w.segment<3>(9) * dt;

  return state + dx;
}
