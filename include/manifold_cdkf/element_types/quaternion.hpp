#pragma once

#include "base.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <limits>

template <typename Scalar>
class QuaternionElement final
    : public ManifoldElement<QuaternionElement<Scalar>, Scalar, 3>
{
 public:
  using Base = ManifoldElement<QuaternionElement<Scalar>, Scalar, 3>;
  using TangentVec = typename Base::TangentVec;

  using Quat = Eigen::Quaternion<Scalar>;
  using ElementType = Quat;

  QuaternionElement(const Quat &q = Quat::Identity()) : quat_{q} {}

  Quat getValue() const { return quat_; }
  void setValue(const Quat &q) { quat_ = q; }

  QuaternionElement operator+(const TangentVec &diff) const override
  {
    return QuaternionElement(quat_ * vec_to_quat(diff));
  }

  TangentVec operator-(const QuaternionElement &q) const override
  {
    return quat_to_vec(q.getValue().conjugate() * quat_);
  }

  static TangentVec angle_axis_to_vec(const Scalar &angle,
                                      const Eigen::Matrix<Scalar, 3, 1> &axis)
  {
    return axis * angle;
  }

  friend std::ostream &operator<<(std::ostream &stream,
                                  const QuaternionElement &element)
  {
    stream << element.quat_.coeffs().transpose();
    return stream;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  static TangentVec quat_to_vec(const Quat &q)
  {
    const auto q_vec_norm = q.vec().norm();
    if(q_vec_norm < 10 * std::numeric_limits<Scalar>::epsilon())
      return TangentVec::Zero();
    else
      return 2 * std::atan2(q_vec_norm, q.w()) * q.vec() / q_vec_norm;
  }

  static Quat vec_to_quat(const TangentVec &vec)
  {
    const auto v_norm = vec.norm();
    if(v_norm < 10 * std::numeric_limits<Scalar>::epsilon())
      return Quat::Identity();

    Quat q;
    q.w() = std::cos(v_norm / 2);
    q.vec() = std::sin(v_norm / 2) * vec / v_norm;
    return q;
  }

  Quat quat_;
};
