#pragma once

#include "compound.hpp"
#include "vector.hpp"
#include "quaternion.hpp"

/**
 * CompoundElement containing [Position, Orientation, Linear Velocity]
 */
template <typename Scalar>
class PoseVelElement final
    : public CompoundElement<Scalar, VectorElement<Scalar, 3>,
                             QuaternionElement<Scalar>,
                             VectorElement<Scalar, 3>>
{
 public:
  using Base =
      CompoundElement<Scalar, VectorElement<Scalar, 3>,
                      QuaternionElement<Scalar>, VectorElement<Scalar, 3>>;
  using TangentVec = typename Base::TangentVec;

  template <int N>
  using Vec = Eigen::Matrix<Scalar, N, 1>;
  using Quat = Eigen::Quaternion<Scalar>;

  PoseVelElement(Base const &b) : Base(b) {}

  explicit PoseVelElement(const Vec<3> &position = Vec<3>::Zero(),
                          const Quat &orientation = Quat::Identity(),
                          const Vec<3> &velocity = Vec<3>::Zero())
      : Base(position, orientation, velocity)
  {
  }

  Vec<3> getPosition() const { return Base::template getValue<0>(); }
  void setPosition(const Vec<3> &v) { Base::template setValue<0>(v); }

  Quat getOrientation() const { return Base::template getValue<1>(); }
  void setOrientation(const Quat &q) { Base::template setValue<1>(q); }

  Vec<3> getVelocity() const { return Base::template getValue<2>(); }
  void setVelocity(const Vec<3> &v) { Base::template setValue<2>(v); }

  friend std::ostream &operator<<(std::ostream &stream, PoseVelElement const &e)
  {
    stream << "Position: " << e.template get<0>() << "\n"
           << "Orientation: " << e.template get<1>() << "\n"
           << "Velocity: " << e.template get<2>();
    return stream;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
