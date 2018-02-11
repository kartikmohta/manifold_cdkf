#pragma once

#include <Eigen/Core>

template <typename Derived, typename Scalar, unsigned int tangent_dim>
class ManifoldElement
{
 public:
  using Scalar_t = Scalar;
  using TangentVec = Eigen::Matrix<Scalar, tangent_dim, 1>;

  virtual ~ManifoldElement() = default;
  virtual Derived operator+(const TangentVec &diff) const = 0;
  virtual TangentVec operator-(const Derived &element) const = 0;

  static constexpr unsigned int tangent_dim_ = tangent_dim;
};
