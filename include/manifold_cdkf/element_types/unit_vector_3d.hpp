#pragma once

#include "base.hpp"

template <typename Scalar>
class UnitVector3DElement final
    : public ManifoldElement<UnitVector3DElement<Scalar>, Scalar, 2>
{
 public:
  using Base = ManifoldElement<UnitVector3DElement<Scalar>, Scalar, 2>;
  using TangentVec = typename Base::TangentVec;

  using UnitVec = Eigen::Matrix<Scalar, 3, 1>;
  using ElementType = UnitVec;

  UnitVector3DElement(const UnitVec &v = UnitVec::UnitX())
  {
    vec_ = v.normalized();
  }

  UnitVec const &getValue() const { return vec_; }
  void setValue(const UnitVec &v) { vec_ = v.normalized(); }

  UnitVector3DElement operator+(const TangentVec &diff) const override
  {
    return UnitVector3DElement(this->R(vec_) * this->exp(diff));
  }

  TangentVec operator-(const UnitVector3DElement &v) const override
  {
    return this->log(this->R(v.getValue()).transpose() * vec_);
  }

  friend std::ostream &operator<<(std::ostream &stream,
                                  const UnitVector3DElement &element)
  {
    stream << element.vec_.transpose();
    return stream;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  static Eigen::Matrix<Scalar, 3, 3> R(UnitVec const &x)
  {
    Scalar const alpha = std::atan2(x(2), x(1));
    Scalar const c = std::cos(alpha), s = std::sin(alpha);
    Scalar const r = std::hypot(x(1), x(2));
    return (Eigen::Matrix<Scalar, 3, 3>() << x(0), -r, 0, x(1), x(0) * c, -s,
            x(2), x(0) * s, c)
        .finished();
  }

  static UnitVec exp(TangentVec const &delta)
  {
    Scalar const delta_norm = delta.norm();
    if(delta_norm < 10 * std::numeric_limits<Scalar>::epsilon())
      return UnitVec::UnitX();
    return (UnitVec() << std::cos(delta_norm),
            std::sin(delta_norm) * delta(0) / delta_norm,
            std::sin(delta_norm) * delta(1) / delta_norm)
        .finished();
  }

  static TangentVec log(UnitVec const &x)
  {
    Scalar const w = x(0);
    TangentVec const v = x.template tail<2>();
    Scalar const v_norm = v.norm();

    if(v_norm < 10 * std::numeric_limits<Scalar>::epsilon())
      return std::atan2(0, w) * TangentVec::UnitX();
    else
      return std::atan2(v_norm, w) * v / v_norm;
  }

  UnitVec vec_;
};
