#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <random>

#include "manifold_cdkf/element_types/so2.hpp"

using Scalar = double;

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<Scalar> dis(-M_PI, M_PI);
// Note that we want numbers from (-M_PI, M_PI] hence we use -dis(gen) to get the random number

static auto const elem_zero = SO2Element<Scalar>(0);
static auto const elem_pi_2 = SO2Element<Scalar>(M_PI / 2);
static auto const elem_pi = SO2Element<Scalar>(M_PI);
static auto const elem_m_pi_2 = SO2Element<Scalar>(-M_PI / 2);
static auto const elem_m_pi = SO2Element<Scalar>(-M_PI);

TEST_CASE("x + (y - x) == y")
{
  for(int i = 0; i < 1000; ++i)
  {
    auto const x = SO2Element<Scalar>(-dis(gen));
    auto const y = SO2Element<Scalar>(-dis(gen));
    CHECK_EQ((x + (y - x)).getValue(), doctest::Approx(y.getValue()));
  }
}

TEST_CASE("(x + d) - x == d")
{
  for(int i = 0; i < 1000; ++i)
  {
    auto const x = SO2Element<Scalar>(-dis(gen));
    auto const d = SO2Element<Scalar>::TangentVec(-dis(gen));

    CHECK_EQ(((x + d) - x).value(), doctest::Approx(d.value()));
  }
}

TEST_CASE("||(x + d1) - (x + d2)|| <= ||d1 - d2||")
{
  for(int i = 0; i < 1000; ++i)
  {
    auto const x = SO2Element<Scalar>(-dis(gen));
    auto const d1 = SO2Element<Scalar>::TangentVec(-dis(gen));
    auto const d2 = SO2Element<Scalar>::TangentVec(-dis(gen));

    CHECK_LE(((x + d1) - (x + d2)).norm(), doctest::Approx((d1 - d2).norm()));
  }
}
