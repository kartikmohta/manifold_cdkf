#pragma once

#include "element_types/scalar.hpp"
#include "element_types/pose.hpp"
#include "element_types/pose_vel.hpp"

template <typename State>
PoseElement<typename State::Scalar_t> MeasurementModelPose(State const &state)
{
  return PoseElement<typename State::Scalar_t>(state.getPosition(),
                                               state.getOrientation());
}

template <typename State>
ScalarElement<typename State::Scalar_t> MeasurementModelHeight(
    State const &state)
{
  const auto q = state.getOrientation();
  return state.getPosition()(2) / (1 - 2 * (q.x() * q.x() + q.y() * q.y()));
}

template <typename State>
PoseVelElement<typename State::Scalar_t> MeasurementModelPoseVel(
    State const &state)
{
  return PoseVelElement<typename State::Scalar_t>(
      state.getPosition(), state.getOrientation(), state.getVelocity());
}
