#pragma once

#include "compound.hpp"
#include "quaternion.hpp"
#include "vector.hpp"

/**
 * CompoundElement containing [Position, Orientation]
 */
template <typename Scalar>
class PoseElement final : public CompoundElement<VectorElement<Scalar, 3>,
                                                 QuaternionElement<Scalar>>
{
 public:
  using Base =
      CompoundElement<VectorElement<Scalar, 3>, QuaternionElement<Scalar>>;
  using TangentVec = typename Base::TangentVec;

  template <int N>
  using Vec = Eigen::Matrix<Scalar, N, 1>;
  using Quat = Eigen::Quaternion<Scalar>;

  PoseElement(Base const &b) : Base(b) {}

  explicit PoseElement(const Vec<3> &position = Vec<3>::Zero(),
                       const Quat &orientation = Quat::Identity())
      : Base(position, orientation)
  {
  }

  Vec<3> const &getPosition() const { return Base::template getValue<0>(); }
  void setPosition(const Vec<3> &v) { Base::template setValue<0>(v); }

  Quat const &getOrientation() const { return Base::template getValue<1>(); }
  void setOrientation(const Quat &q) { Base::template setValue<1>(q); }

  friend std::ostream &operator<<(std::ostream &stream, PoseElement const &e)
  {
    stream << "Position: " << e.template get<0>() << "\n"
           << "Orientation: " << e.template get<1>();
    return stream;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
