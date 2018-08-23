#include <iostream>

#include "manifold_cdkf/element_types/pose.hpp"
#include "manifold_cdkf/element_types/pose_vel.hpp"
#include "manifold_cdkf/manifold_cdkf.hpp"
#include "manifold_cdkf/measurement_funcs.hpp"

using Scalar_t = double;

using State = PoseVelElement<Scalar_t>;
using StateCov =
    Eigen::Matrix<State::Scalar_t, State::tangent_dim_, State::tangent_dim_>;

template <int N>
using Vec = Eigen::Matrix<State::Scalar_t, N, 1>;

template <int N>
using SquareMat = Eigen::Matrix<State::Scalar_t, N, N>;

using Pose = PoseElement<Scalar_t>;

static State processModelConstantVelocity(State const &x, Vec<0> const &u,
                                          Vec<3> const &w, double dt)
{
  auto dx = State::TangentVec::Zero().eval();
  dx.segment<3>(0) = x.getVelocity() * dt;
  dx.segment<3>(3) = w * dt;
  return x + dx;
}

int main(int argc, char *argv[])
{
  // Construct the filter with an initial state, initial covariance and process
  // model
  auto const x0 =
      State{Vec<3>::Zero(), State::Quat::Identity(), Vec<3>::Zero()};
  auto const P0 = (0.1 * StateCov::Identity()).eval();
  auto const process_model =
      std::function<State(State const &, Vec<0> const &, Vec<3> const &,
                          double)>(&processModelConstantVelocity);
  auto cdkf = ManifoldCDKF<State, Vec<0>, Vec<3>>{x0, P0, process_model};

  std::cout << cdkf.getState() << "\n";

  // Apply a few process updates (predictions)
  auto const Q = (0.25 * SquareMat<3>::Identity()).eval();

  if(cdkf.processUpdate(0.1, Vec<0>{}, Q))
    std::cout << cdkf.getState() << "\n";

  if(cdkf.processUpdate(0.1, Vec<0>{}, Q))
    std::cout << cdkf.getState() << "\n";

  if(cdkf.processUpdate(0.1, Vec<0>{}, Q))
    std::cout << cdkf.getState() << "\n";

  // Measurement update
  {
    auto const new_position = Vec<3>{1, 2, 3};
    auto const new_orientation =
        State::Quat{Eigen::AngleAxis<Scalar_t>(M_PI / 4, Vec<3>::UnitX())};
    auto const z = Pose{new_position, new_orientation};
    auto const R = SquareMat<z.tangent_dim_>::Identity().eval();

    cdkf.measurementUpdate(&MeasurementModelPose<State>, z, R, false);
    std::cout << cdkf.getState() << "\n";
  }

  // Measurement update with zero covariance
  {
    auto const new_position = Vec<3>{2, 3, 1};
    auto const new_orientation =
        State::Quat{Eigen::AngleAxis<Scalar_t>(M_PI / 2, Vec<3>::UnitX())};
    auto z = Pose{new_position, new_orientation};
    auto const R = SquareMat<z.tangent_dim_>::Zero().eval();

    cdkf.measurementUpdate(&MeasurementModelPose<State>, z, R, false);
    std::cout << cdkf.getState() << "\n";
  }

  return 0;
}
