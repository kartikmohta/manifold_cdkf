#include "base.hpp"

template <typename Scalar>
class SO2Element final : public ManifoldElement<SO2Element<Scalar>, Scalar, 1>
{
 public:
  using Base = ManifoldElement<SO2Element<Scalar>, Scalar, 1>;
  using TangentVec = typename Base::TangentVec;

  using ElementType = Scalar;

  SO2Element(const Scalar &angle = Scalar{0}) : angle_{normalize_angle(angle)}
  {
  }

  Scalar getValue() const { return angle_; }
  void setValue(const Scalar &angle) { angle_ = normalize_angle(angle); }

  SO2Element operator+(const TangentVec &diff) const override
  {
    return SO2Element(normalize_angle(angle_ + diff(0)));
  }

  TangentVec operator-(const SO2Element &other) const override
  {
    return TangentVec(normalize_angle(angle_ - other.angle_));
  }

  friend std::ostream &operator<<(std::ostream &stream,
                                  const SO2Element &element)
  {
    stream << element.angle_;
    return stream;
  }

 private:
  static constexpr Scalar normalize_angle(Scalar angle)
  {
    constexpr Scalar pi = M_PI, two_pi = 2 * M_PI;
    return angle - (std::ceil((angle + pi) / two_pi) - Scalar{1}) * two_pi;
  }

  Scalar angle_;
};
