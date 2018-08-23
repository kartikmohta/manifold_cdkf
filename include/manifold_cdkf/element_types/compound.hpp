#pragma once

#include "base.hpp"

template <typename T>
constexpr T templateSum(T v)
{
  return v;
}

template <typename T, typename... Ts>
constexpr T templateSum(T first, Ts... args)
{
  return first + templateSum(args...);
}

template <typename Scalar, typename... Args>
class CompoundElement
    : public ManifoldElement<CompoundElement<Scalar, Args...>, Scalar,
                             templateSum(Args::tangent_dim_...)>
{
 private:
  template <typename... Ts>
  struct TypeList
  {
    template <std::size_t N>
    using type = typename std::tuple_element<N, std::tuple<Ts...>>::type;
  };

 public:
  using Base = ManifoldElement<CompoundElement<Scalar, Args...>, Scalar,
                               templateSum(Args::tangent_dim_...)>;
  using TangentVec = typename Base::TangentVec;

  CompoundElement() : t_{std::tuple<Args...>()} {}
  explicit CompoundElement(Args const &... vs) : t_{std::make_tuple(vs...)} {}

  template <unsigned int N>
  typename TypeList<Args...>::template type<N>::ElementType getValue() const
  {
    return std::get<N>(t_).getValue();
  }

  template <unsigned int N>
  void setValue(
      typename TypeList<Args...>::template type<N>::ElementType const &v)
  {
    std::get<N>(t_).setValue(v);
  }

  template <unsigned int N>
  typename TypeList<Args...>::template type<N> get() const
  {
    return std::get<N>(t_);
  }

  template <unsigned int N>
  void set(typename TypeList<Args...>::template type<N> const &v)
  {
    std::get<N>(t_) = v;
  }

  CompoundElement operator+(const TangentVec &diff) const override
  {
    CompoundElement p;
    add(diff, p);
    return p;
  }

  TangentVec operator-(CompoundElement const &other) const override
  {
    TangentVec v;
    subtract(other, v);
    return v;
  }

  friend std::ostream &operator<<(std::ostream &stream,
                                  CompoundElement const &element)
  {
    stream << "CompoundElement with " << sizeof...(Args) << " elements"
           << std::endl;
    element.print(stream);
    return stream;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // All these functions are called sort of recursively for each tuple element

  template <size_t i = 0, size_t segment_begin_idx = 0>
  std::enable_if_t<i < sizeof...(Args)> add(TangentVec const &diff,
                                            CompoundElement &p) const
  {
    constexpr size_t tangent_dim =
        TypeList<Args...>::template type<i>::tangent_dim_;

    std::get<i>(p.t_) =
        std::get<i>(t_) + diff.template segment<tangent_dim>(segment_begin_idx);

    add<i + 1, segment_begin_idx + tangent_dim>(diff, p);
  }
  template <size_t i = 0, size_t segment_begin_idx = 0>
  std::enable_if_t<i >= sizeof...(Args)> add(TangentVec const &,
                                             CompoundElement &) const
  {
  }

  template <size_t i = 0, size_t segment_begin_idx = 0>
  std::enable_if_t<i < sizeof...(Args)> subtract(CompoundElement const &p,
                                                 TangentVec &diff) const
  {
    constexpr size_t tangent_dim =
        TypeList<Args...>::template type<i>::tangent_dim_;

    diff.template segment<tangent_dim>(segment_begin_idx) =
        std::get<i>(t_) - std::get<i>(p.t_);

    subtract<i + 1, segment_begin_idx + tangent_dim>(p, diff);
  }
  template <size_t i = 0, size_t segment_begin_idx = 0>
  std::enable_if_t<i >= sizeof...(Args)> subtract(CompoundElement const &,
                                                  TangentVec &) const
  {
  }

  template <size_t i = 0>
  std::enable_if_t<i < sizeof...(Args)> print(std::ostream &stream) const
  {
    stream << "- Element " << i << ": " << std::get<i>(t_) << std::endl;
    print<i + 1>(stream);
  }
  template <size_t i = 0>
  std::enable_if_t<i >= sizeof...(Args)> print(std::ostream &) const
  {
  }

  std::tuple<Args...> t_;
};
