#pragma once

#include "base.hpp"

template <typename Scalar, int N>
class VectorElement final
    : public ManifoldElement<VectorElement<Scalar, N>, Scalar, N>
{
 public:
  using Base = ManifoldElement<VectorElement<Scalar, N>, Scalar, N>;
  using TangentVec = typename Base::TangentVec;

  using Vec = Eigen::Matrix<Scalar, N, 1>;
  using ElementType = Vec;

  VectorElement(const Vec &v = Vec::Zero()) : vec_{v} {}

  Vec const &getValue() const { return vec_; }
  void setValue(const Vec &v) { vec_ = v; }

  VectorElement operator+(const TangentVec &diff) const override
  {
    return VectorElement(vec_ + diff);
  }

  TangentVec operator-(const VectorElement &v) const override
  {
    return (vec_ - v.getValue());
  }

  friend std::ostream &operator<<(std::ostream &stream,
                                  const VectorElement &element)
  {
    stream << element.vec_.transpose();
    return stream;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  Vec vec_;
};
