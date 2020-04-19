#pragma once

#include "base.hpp"

template <typename Scalar, typename T, unsigned int N>
class ArrayElement : public ManifoldElement<ArrayElement<Scalar, T, N>, Scalar,
                                            N * T::tangent_dim_>
{
 public:
  using Base =
      ManifoldElement<ArrayElement<Scalar, T, N>, Scalar, N * T::tangent_dim_>;
  using TangentVec = typename Base::TangentVec;

  ArrayElement() = default;
  explicit ArrayElement(std::array<T, N> const &array) : array_{array} {}

  template <unsigned int M>
  typename T::ElementType const &getValue() const
  {
    return array_[M].getValue();
  }

  typename T::ElementType const &getValue(unsigned int M) const
  {
    return array_[M].getValue();
  }

  template <unsigned int M>
  void setValue(typename T::ElementType const &v)
  {
    array_[M].setValue(v);
  }

  void setValue(unsigned int M, typename T::ElementType const &v)
  {
    array_[M].setValue(v);
  }

  template <unsigned int M>
  T get() const
  {
    return array_[M];
  }

  T get(unsigned int M) const { return array_[M]; }

  template <unsigned int M>
  void set(T const &v)
  {
    array_[M] = v;
  }

  void set(unsigned int M, T const &v) { array_[M] = v; }

  ArrayElement operator+(const TangentVec &diff) const override
  {
    ArrayElement p;
    for(unsigned int i = 0; i < N; ++i)
    {
      p.array_[i] = array_[i] +
                    diff.template segment<T::tangent_dim_>(i * T::tangent_dim_);
    }
    return p;
  }

  TangentVec operator-(ArrayElement const &other) const override
  {
    TangentVec v;
    for(unsigned int i = 0; i < N; ++i)
    {
      v.template segment<T::tangent_dim_>(i * T::tangent_dim_) =
          array_[i] - other.array_[i];
    }
    return v;
  }

  friend std::ostream &operator<<(std::ostream &stream,
                                  ArrayElement const &element)
  {
    stream << "ArrayElement with " << N << " elements"
           << "\n";
    for(unsigned int i = 0; i < N; ++i)
    {
      stream << "- [" << i << "] : " << element.array_[i] << "\n";
    }
    return stream;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  std::array<T, N> array_;
};
