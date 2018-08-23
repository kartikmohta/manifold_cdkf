#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "manifold_cdkf/element_types/unit_vector_3d.hpp"

int main(int argc, char *argv[])
{
  std::srand(std::time(nullptr));
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  int res = context.run();
  return res;
}

using Scalar_t = double;

template <int N>
using Vec = Eigen::Matrix<Scalar_t, N, 1>;

static auto const vx = Vec<3>::UnitX().eval();
static auto const vy = Vec<3>::UnitY().eval();
static auto const vz = Vec<3>::UnitZ().eval();
static auto const elem_xp = UnitVector3DElement<Scalar_t>(vx);
static auto const elem_xn = UnitVector3DElement<Scalar_t>(-vx);
static auto const elem_yp = UnitVector3DElement<Scalar_t>(vy);
static auto const elem_yn = UnitVector3DElement<Scalar_t>(-vy);
static auto const elem_zp = UnitVector3DElement<Scalar_t>(vz);
static auto const elem_zn = UnitVector3DElement<Scalar_t>(-vz);

TEST_CASE("x + (y - x) == y")
{
  for(int i = 0; i < 1000; ++i)
  {
    auto const v_random = Vec<3>::Random().normalized();
    auto const elem_random = UnitVector3DElement<Scalar_t>(v_random);

    CHECK((elem_random + (elem_xn - elem_random)).getValue().isApprox(elem_xn.getValue()));
    CHECK((elem_random + (elem_yn - elem_random)).getValue().isApprox(elem_yn.getValue()));
    CHECK((elem_random + (elem_zn - elem_random)).getValue().isApprox(elem_zn.getValue()));
    CHECK((elem_random + (elem_xp - elem_random)).getValue().isApprox(elem_xp.getValue()));
    CHECK((elem_random + (elem_yp - elem_random)).getValue().isApprox(elem_yp.getValue()));
    CHECK((elem_random + (elem_zp - elem_random)).getValue().isApprox(elem_zp.getValue()));
    CHECK((elem_xp + (elem_random - elem_xp)).getValue().isApprox(elem_random.getValue()));
    CHECK((elem_yp + (elem_random - elem_yp)).getValue().isApprox(elem_random.getValue()));
    CHECK((elem_zp + (elem_random - elem_zp)).getValue().isApprox(elem_random.getValue()));
    CHECK((elem_xn + (elem_random - elem_xn)).getValue().isApprox(elem_random.getValue()));
    CHECK((elem_yn + (elem_random - elem_yn)).getValue().isApprox(elem_random.getValue()));
    CHECK((elem_zn + (elem_random - elem_zn)).getValue().isApprox(elem_random.getValue()));
  }
}

TEST_CASE("(x + d) - x == d")
{
  for(int i = 0; i < 1000; ++i)
  {
    auto const v_random = Vec<3>::Random().normalized();
    auto const elem_random = UnitVector3DElement<Scalar_t>(v_random);
    auto const d = UnitVector3DElement<Scalar_t>::TangentVec::Random().eval();

    CHECK(((elem_xp + d) - elem_xp).isApprox(d));
    CHECK(((elem_yp + d) - elem_yp).isApprox(d));
    CHECK(((elem_zp + d) - elem_zp).isApprox(d));
    CHECK(((elem_xn + d) - elem_xn).isApprox(d));
    CHECK(((elem_yn + d) - elem_yn).isApprox(d));
    CHECK(((elem_zn + d) - elem_zn).isApprox(d));
    CHECK(((elem_random + d) - elem_random).isApprox(d));
  }
}

TEST_CASE("||(x + d1) - (x + d2)|| < ||d1 - d2||")
{
  for(int i = 0; i < 1000; ++i)
  {
    auto const v_random = Vec<3>::Random().normalized();
    auto const elem_random = UnitVector3DElement<Scalar_t>(v_random);
    auto const d1 = UnitVector3DElement<Scalar_t>::TangentVec::Random().eval();
    auto const d2 = UnitVector3DElement<Scalar_t>::TangentVec::Random().eval();

    CHECK(((elem_random + d1) - (elem_random + -d1)).norm() <= doctest::Approx((d1 - -d1).norm()));
    CHECK(((elem_random + d2) - (elem_random + -d2)).norm() <= doctest::Approx((d2 - -d2).norm()));
    CHECK(((elem_random + d1) - (elem_random + d2)).norm() <= doctest::Approx((d1 - d2).norm()));
    CHECK(((elem_random + d2) - (elem_random + d1)).norm() <= doctest::Approx((d2 - d1).norm()));
  }
}

TEST_CASE("90 deg rotation: X,Y")
{
  CHECK((elem_yp - elem_xp).isApprox((Vec<2>() << M_PI_2, 0).finished()));
  CHECK((elem_xp - elem_yp).isApprox((Vec<2>() << -M_PI_2, 0).finished()));
  CHECK((elem_yp - elem_xn).isApprox((Vec<2>() << M_PI_2, 0).finished()));
  CHECK((elem_xn - elem_yp).isApprox((Vec<2>() << M_PI_2, 0).finished()));
  CHECK((elem_yn - elem_xp).isApprox((Vec<2>() << -M_PI_2, 0).finished()));
  CHECK((elem_xp - elem_yn).isApprox((Vec<2>() << -M_PI_2, 0).finished()));
  CHECK((elem_yn - elem_xn).isApprox((Vec<2>() << -M_PI_2, 0).finished()));
  CHECK((elem_xn - elem_yn).isApprox((Vec<2>() << M_PI_2, 0).finished()));
}

TEST_CASE("90 deg rotation: Y,Z")
{
  CHECK((elem_zp - elem_yp).isApprox((Vec<2>() << 0, M_PI_2).finished()));
  CHECK((elem_yp - elem_zp).isApprox((Vec<2>() << 0, -M_PI_2).finished()));
  CHECK((elem_zp - elem_yn).isApprox((Vec<2>() << 0, -M_PI_2).finished()));
  CHECK((elem_yn - elem_zp).isApprox((Vec<2>() << 0, M_PI_2).finished()));
  CHECK((elem_zn - elem_yp).isApprox((Vec<2>() << 0, -M_PI_2).finished()));
  CHECK((elem_yp - elem_zn).isApprox((Vec<2>() << 0, M_PI_2).finished()));
  CHECK((elem_zn - elem_yn).isApprox((Vec<2>() << 0, M_PI_2).finished()));
  CHECK((elem_yn - elem_zn).isApprox((Vec<2>() << 0, -M_PI_2).finished()));
}

TEST_CASE("90 deg rotation: Z,X")
{
  CHECK((elem_xp - elem_zp).isApprox((Vec<2>() << -M_PI_2, 0).finished()));
  CHECK((elem_zp - elem_xp).isApprox((Vec<2>() << 0, M_PI_2).finished()));
  CHECK((elem_xp - elem_zn).isApprox((Vec<2>() << -M_PI_2, 0).finished()));
  CHECK((elem_zn - elem_xp).isApprox((Vec<2>() << 0, -M_PI_2).finished()));
  CHECK((elem_xn - elem_zp).isApprox((Vec<2>() << M_PI_2, 0).finished()));
  CHECK((elem_zp - elem_xn).isApprox((Vec<2>() << 0, -M_PI_2).finished()));
  CHECK((elem_xn - elem_zn).isApprox((Vec<2>() << M_PI_2, 0).finished()));
  CHECK((elem_zn - elem_xn).isApprox((Vec<2>() << 0, M_PI_2).finished()));
}

TEST_CASE("180 deg rotation: X")
{
  auto const v_pi = (Vec<2>() << M_PI, 0).finished();
  CHECK((elem_xp - elem_xn).isApprox(v_pi));
  CHECK((elem_xn - elem_xp).isApprox(v_pi));
  CHECK((elem_xp + v_pi).getValue().isApprox(elem_xn.getValue()));
  CHECK((elem_xn + v_pi).getValue().isApprox(elem_xp.getValue()));
  CHECK((elem_xp + -v_pi).getValue().isApprox(elem_xn.getValue()));
  CHECK((elem_xn + -v_pi).getValue().isApprox(elem_xp.getValue()));
}

TEST_CASE("180 deg rotation: Y")
{
  auto const v_pi = (Vec<2>() << M_PI, 0).finished();
  CHECK((elem_yp - elem_yn).isApprox(v_pi));
  CHECK((elem_yn - elem_yp).isApprox(v_pi));
  CHECK((elem_yp + v_pi).getValue().isApprox(elem_yn.getValue()));
  CHECK((elem_yn + v_pi).getValue().isApprox(elem_yp.getValue()));
  CHECK((elem_yp + -v_pi).getValue().isApprox(elem_yn.getValue()));
  CHECK((elem_yn + -v_pi).getValue().isApprox(elem_yp.getValue()));
}

TEST_CASE("180 deg rotation: Z")
{
  auto const v_pi = (Vec<2>() << M_PI, 0).finished();
  CHECK((elem_zp - elem_zn).isApprox(v_pi));
  CHECK((elem_zn - elem_zp).isApprox(v_pi));
  CHECK((elem_zp + v_pi).getValue().isApprox(elem_zn.getValue()));
  CHECK((elem_zn + v_pi).getValue().isApprox(elem_zp.getValue()));
  CHECK((elem_zp + -v_pi).getValue().isApprox(elem_zn.getValue()));
  CHECK((elem_zn + -v_pi).getValue().isApprox(elem_zp.getValue()));
}
