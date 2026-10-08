// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./simd_test_common.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

namespace {

  // Empty inputs, exact batches, and tails for the active architecture.
  template <typename T>
  auto batch_sizes() {
    constexpr long w = simd_width<T>();
    return std::array<long, 7>{0, 1, w - 1, w, w + 1, 2 * w + 1, 4 * w + 1};
  }

  template <typename T>
  void expect_close(T actual, T expected) {
    using std::abs;
    constexpr double tolerance = std::same_as<nda::remove_complex_t<T>, float> ? 2.e-5 : 1.e-12;
    EXPECT_LE(abs(actual - expected), tolerance);
  }

  template <typename E, typename A, typename F>
  void check_expression(E const &expression, A const &input, F scalar) {
    nda::array<nda::get_value_t<E>, 1> result = expression;
    for (long i = 0; i < input.size(); ++i) { expect_close(result(i), scalar(input(i))); }
  }

} // namespace

template <typename T>
class SIMDMath : public ::testing::Test {};
using math_types = ::testing::Types<float, double, std::complex<float>, std::complex<double>, long double>;
TYPED_TEST_SUITE(SIMDMath, math_types);

TYPED_TEST(SIMDMath, Functions) {
  using T      = TypeParam;
  using real_t = nda::remove_complex_t<T>;
  for (long n : batch_sizes<T>()) {
    // a stays inside every domain below (acos/asin need [-1, 1], log/sqrt need > 0) with results near 1 for the
    // absolute tolerance; b crosses zero and integers, which abs/floor/conj/max/min need to show any effect
    nda::array<T, 1> a(n), b(n);
    fill_random(a, 1, 0.1, 0.9);
    fill_random(b, 2, -5.0, 5.0);

    // full batches and tails against the standard scalar functions
#define CHECK_MATH(name) check_expression(nda::name(a), a, [](T x) { return std::name(x); })
    CHECK_MATH(exp);
    CHECK_MATH(cos);
    CHECK_MATH(sin);
    CHECK_MATH(tan);
    CHECK_MATH(cosh);
    CHECK_MATH(sinh);
    CHECK_MATH(tanh);
    CHECK_MATH(acos);
    CHECK_MATH(asin);
    CHECK_MATH(atan);
    CHECK_MATH(log);
    CHECK_MATH(sqrt);
#undef CHECK_MATH
    check_expression(nda::abs2(a), a, [](T x) { return std::norm(x); });
    check_expression(nda::reciprocal(a), a, [](T x) { return real_t{1} / x; });
    check_expression(nda::pow(a, real_t{2}), a, [](T x) { return std::pow(x, real_t{2}); });
    check_expression(nda::pow(a, 2), a, [](T x) { return std::pow(x, 2); });

    check_expression(nda::conj(b), b, [](T x) -> T {
      if constexpr (nda::is_complex_v<T>) {
        return std::conj(x);
      } else {
        return x;
      }
    });
    check_expression(nda::real(b), b, [](T x) { return std::real(x); });
    check_expression(nda::imag(b), b, [](T x) { return std::imag(x); });
    check_expression(nda::abs(b), b, [](T x) { return std::abs(x); });
    if constexpr (not nda::is_complex_v<T>) {
      check_expression(nda::floor(b), b, [](T x) { return std::floor(x); });
      check_expression(nda::max(b, T(0.5) * b), b, [](T x) { return std::max(x, T(0.5) * x); });
      check_expression(nda::min(b, T(0.5) * b), b, [](T x) { return std::min(x, T(0.5) * x); });
    }

#ifdef NDA_HAVE_XSIMD
    // the checks above run the native loads whenever T has a batch
    constexpr bool vectorizable = nda::Vectorizable<T>;
    static_assert(nda::is_simd_enabled_v<decltype(nda::exp(a))> == vectorizable);
    static_assert(nda::is_simd_enabled_v<decltype(nda::abs2(a))> == vectorizable);
    static_assert(nda::is_simd_enabled_v<decltype(nda::reciprocal(a))> == vectorizable);
    static_assert(nda::is_simd_enabled_v<decltype(nda::pow(a, real_t{2}))> == vectorizable);
    static_assert(nda::is_simd_enabled_v<decltype(nda::conj(b))> == vectorizable);
    static_assert(nda::is_simd_enabled_v<decltype(nda::real(b))> == vectorizable);
    static_assert(nda::is_simd_enabled_v<decltype(nda::imag(b))> == vectorizable);
    static_assert(nda::is_simd_enabled_v<decltype(nda::abs(b))> == vectorizable);
    if constexpr (not nda::is_complex_v<T>) {
      static_assert(nda::is_simd_enabled_v<decltype(nda::floor(b))> == vectorizable);
      static_assert(nda::is_simd_enabled_v<decltype(nda::max(b, T(0.5) * b))> == vectorizable);
    }
#endif
  }
}

TEST(SIMD, Promotions) {
  nda::array<std::int8_t, 1> bytes(65);
  bytes = std::numeric_limits<std::int8_t>::min();
  auto magnitude = nda::abs(bytes);
  static_assert(std::same_as<nda::get_value_t<decltype(magnitude)>, int>);
  nda::array<int, 1> result = magnitude;
  for (auto x : result) { EXPECT_EQ(x, 128); }

  nda::array<int, 1> integers(65);
  integers = 2;
  check_expression(nda::sqrt(integers), integers, [](int x) { return std::sqrt(x); });
  check_expression(nda::floor(integers), integers, [](int x) { return std::floor(x); });
  check_expression(nda::imag(integers), integers, [](int x) { return std::imag(x); });

  // a double exponent promotes floats to double, which changes the lane width, so it stays scalar
  nda::array<float, 1> floats(65);
  floats = 0.5f;
#ifdef NDA_HAVE_XSIMD
  static_assert(not nda::is_simd_enabled_v<decltype(nda::pow(floats, 2.0))>);
#endif
  check_expression(nda::pow(floats, 2.0), floats, [](float x) { return std::pow(x, 2.0); });
}

TEST(SIMD, Composition) {
  using complex_t = std::complex<double>;
  nda::array<complex_t, 2, nda::F_layout> a(5, 13);
  fill_random(a, 1, 0.1, 0.9);
  // real/imag change complex -> double but keep the SIMD width, so the whole tree vectorizes
  auto expression = nda::max(nda::real(nda::exp(a)), nda::imag(nda::conj(a)));
#ifdef NDA_HAVE_XSIMD
  static_assert(nda::is_simd_enabled_v<decltype(expression)>);
#endif
  nda::array<double, 2, nda::F_layout> result = expression;
  for (long i = 0; i < 5; ++i) {
    for (long j = 0; j < 13; ++j) { expect_close(result(i, j), std::max(std::real(std::exp(a(i, j))), std::imag(std::conj(a(i, j))))); }
  }
}

TEST(SIMD, Isnan) {
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  nda::array<std::complex<double>, 2> a(5, 13);
  a       = std::complex<double>(0.25, 0.125);
  a(1, 2) = {nan, 0};
  a(2, 4) = {0, nan};
  nda::array<bool, 2> nan_result = nda::isnan(a);
  EXPECT_TRUE(nan_result(1, 2) and nan_result(2, 4));
  EXPECT_EQ(std::count(nan_result.begin(), nan_result.end(), true), 2);
}

#ifdef NDA_HAVE_XSIMD
namespace {
  template <typename F, typename T>
  concept unary_loadable = requires(F const &f, T const &x) { f.load(x); };
} // namespace

TEST(SIMD, LoadConstraints) {
  // a load takes native_simd<T> batches only, so a scalar argument is rejected by the signature
  static_assert(not unary_loadable<nda::detail::conj_f, double>);
  static_assert(not unary_loadable<nda::detail::real_f, std::complex<double>>);
  // same-type loads that must be rejected by the signature (unsupported value types or domains)
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::max_f, std::complex<double>, std::complex<double>, std::complex<double>>);
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::min_f, std::complex<float>, std::complex<float>, std::complex<float>>);
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::conj_f, bool, bool>);
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::conj_f, long double, long double>);
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::imag_f, int, int>);
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::reciprocal_f<>, float, float>);
  static_assert(nda::LoadWithNativeSimdTo<nda::detail::reciprocal_f<float>, float, float>);
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::pow_f<>, float, float>);
  static_assert(nda::LoadWithNativeSimdTo<nda::detail::pow_f<float>, float, float>);
  static_assert(nda::LoadWithNativeSimdTo<nda::detail::pow_f<int>, double, double>);
  static_assert(nda::LoadWithNativeSimdTo<nda::detail::abs2_f, float, float>);
  static_assert(nda::LoadWithNativeSimdTo<nda::detail::abs2_f, float, std::complex<float>>);

  // type-changing loads: complex batch in, real batch of the same width out
  static_assert(nda::LoadWithNativeSimdTo<nda::detail::abs_f, double, std::complex<double>>);
  static_assert(nda::LoadWithNativeSimdTo<nda::detail::abs_f, float, std::complex<float>>);
  static_assert(nda::LoadWithNativeSimdTo<nda::detail::abs_f, double, double>);
  // Same-type loads must also agree with the scalar result, which promotes narrow signed integers to int.
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::abs_f, std::int8_t, std::int8_t>);
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::abs_f, std::int16_t, std::int16_t>);
  static_assert(nda::LoadWithNativeSimdTo<nda::detail::abs_f, int, int>);
  static_assert(nda::LoadWithNativeSimdTo<nda::detail::real_f, double, std::complex<double>>);
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::real_f, std::complex<double>, std::complex<double>>);
  static_assert(nda::LoadWithNativeSimdTo<nda::detail::max_f, double, double, double>);
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::max_f, double, std::complex<double>, std::complex<double>>);

  // A load that changes the SIMD width (double -> float batches) must not be dispatched natively.
  struct narrowing_f {
    float operator()(double x) const { return static_cast<float>(x); }
    [[nodiscard]] nda::native_simd<float> load(nda::native_simd<double> const &) const { return {0.f}; }
  };
  static_assert(nda::LoadWithNativeSimdTo<narrowing_f, float, double>);
  using narrowing_expr = decltype(nda::map(narrowing_f{})(std::declval<nda::array<double, 1> &>()));
  static_assert(std::same_as<nda::get_value_t<narrowing_expr>, float>);
  static_assert(not nda::is_simd_enabled_v<narrowing_expr>);

  // These probes must reject the signature without instantiating an unsupported mathematical body.
  static_assert(not unary_loadable<nda::detail::floor_f, nda::native_simd<std::complex<double>>>);
  static_assert(not unary_loadable<nda::detail::exp_f, nda::native_simd<int>>);
  static_assert(not unary_loadable<nda::detail::abs_f, nda::native_simd<unsigned int>>);
  static_assert(nda::LoadWithNativeSimdTo<nda::detail::exp_f, double, double>);
}
#endif
