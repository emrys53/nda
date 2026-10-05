// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

TEST(SIMD, Domains) {
  static_assert(not nda::SimdArithmetic<bool>);
  static_assert(not nda::SimdRealOrComplex<long double>);
  static_assert(not nda::SimdPreservesLaneWidth<double, long double>);
  static_assert(nda::PreservesScalarType<nda::detail::abs_f, double>);
  static_assert(not nda::PreservesScalarType<nda::detail::abs_f, std::int8_t>);
  static_assert(not nda::PreservesScalarType<nda::detail::abs_f, std::complex<double>>);
#ifndef NDA_HAVE_XSIMD
  static_assert(not nda::SimdArithmetic<int>);
  static_assert(not nda::SimdSigned<double>);
  static_assert(not nda::SimdReal<float>);
  static_assert(not nda::SimdRealOrComplex<std::complex<double>>);
  static_assert(not nda::SimdComplex<std::complex<float>>);
  static_assert(not nda::SimdPreservesLaneWidth<double, std::complex<double>>);
#endif
}

namespace {

  // Empty inputs, exact batches, and tails for the active architecture.
  template <typename T>
  auto batch_sizes() {
    constexpr long width = [] {
#ifdef NDA_HAVE_XSIMD
      if constexpr (nda::Vectorizable<T>) { return long(nda::native_simd<T>::size); }
#endif
      return 8L;
    }();
    return std::array<long, 7>{0, 1, width - 1, width, width + 1, 2 * width + 1, 4 * width + 1};
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

  template <typename T>
  void check_math() {
    for (long n : batch_sizes<T>()) {
      nda::array<T, 1> a(n);
      for (long i = 0; i < n; ++i) {
        using real_t = nda::remove_complex_t<T>;
        if constexpr (nda::is_complex_v<T>) {
          a(i) = T(real_t(0.25 + 0.001 * i), real_t(0.125));
        } else {
          a(i) = T(0.25 + 0.001 * i);
        }
      }

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

#ifdef NDA_HAVE_XSIMD
      static_assert(nda::is_simd_enabled_v<decltype(nda::exp(a))> == nda::Vectorizable<T>);
#endif
    }
  }

} // namespace

TEST(SIMD, Math) {
  check_math<float>();
  check_math<double>();
  check_math<std::complex<float>>();
  check_math<std::complex<double>>();
  check_math<long double>();
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

  nda::array<float, 1> floats(65);
  floats = 0.5f;
  check_expression(nda::abs2(floats), floats, [](float x) { return nda::detail::abs2(x); });
  check_expression(nda::reciprocal(floats), floats, [](float x) { return 1.0 / x; });
  check_expression(nda::pow(floats, 2.0), floats, [](float x) { return std::pow(x, 2.0); });
}

TEST(SIMD, Composition) {
  using complex_t = std::complex<double>;
  nda::array<complex_t, 2, nda::F_layout> a(5, 13);
  a = complex_t(0.25, 0.125);
  // real/imag change complex -> double but keep the SIMD width, so the whole tree vectorizes
  auto expression = nda::max(nda::real(nda::exp(a)), nda::imag(nda::conj(a)));
#ifdef NDA_HAVE_XSIMD
  static_assert(nda::is_simd_enabled_v<decltype(expression)>);
#endif
  nda::array<double, 2, nda::F_layout> result = expression;
  for (auto x : result) { expect_close(x, std::max(std::real(std::exp(a(0, 0))), std::imag(std::conj(a(0, 0))))); }
}

TEST(SIMD, Isnan) {
  using complex_t = std::complex<double>;
  nda::array<complex_t, 2> a(5, 13);
  a = complex_t(0.25, 0.125);
  nda::array<bool, 2> nan_result = nda::isnan(a);
  for (auto x : nan_result) { EXPECT_FALSE(x); }
  a(2, 4) = complex_t(0, std::numeric_limits<double>::quiet_NaN());
  nan_result = nda::isnan(a);
  EXPECT_TRUE(nan_result(2, 4));
  EXPECT_FALSE(nan_result(0, 0));
}

namespace {

  template <typename T>
  void check_complex_to_real() {
    using real_t = nda::remove_complex_t<T>;
    for (long n : batch_sizes<T>()) {
      nda::array<T, 1> y(n), z(n);
      for (long i = 0; i < n; ++i) {
        y(i) = T(real_t(0.25 + 0.001 * i), real_t(-0.125 + 0.002 * i));
        z(i) = T(real_t(-0.5 + 0.003 * i), real_t(0.75));
      }

      check_expression(nda::abs(y), y, [](T x) { return std::abs(x); });
      check_expression(nda::real(y), y, [](T x) { return std::real(x); });
      check_expression(nda::imag(y), y, [](T x) { return std::imag(x); });
      if constexpr (std::same_as<T, std::complex<double>>) { check_expression(nda::abs2(y), y, [](T x) { return std::norm(x); }); }

      nda::array<real_t, 1> sum = nda::abs(y) + nda::abs(z);
      for (long i = 0; i < n; ++i) { expect_close(sum(i), std::abs(y(i)) + std::abs(z(i))); }

#ifdef NDA_HAVE_XSIMD
      static_assert(nda::is_simd_enabled_v<decltype(nda::abs(y))>);
      static_assert(nda::is_simd_enabled_v<decltype(nda::real(y))>);
      static_assert(nda::is_simd_enabled_v<decltype(nda::imag(y))>);
      static_assert(nda::is_simd_enabled_v<decltype(nda::abs(y) + nda::abs(z))>);
      static_assert(nda::is_simd_enabled_v<decltype(nda::abs(y) * nda::real(z) - real_t{1})>);
      // abs2's SIMD load is double only
      static_assert(nda::is_simd_enabled_v<decltype(nda::abs2(y))> == std::same_as<T, std::complex<double>>);
#endif
    }
  }

} // namespace

TEST(SIMD, ComplexToReal) {
  check_complex_to_real<std::complex<float>>();
  check_complex_to_real<std::complex<double>>();
}

#ifdef NDA_HAVE_XSIMD
TEST(SIMD, NativeDomains) {
  static_assert(nda::SimdArithmetic<int>);
  static_assert(nda::SimdSigned<double>);
  static_assert(not nda::SimdSigned<unsigned int>);
  static_assert(nda::SimdReal<float>);
  static_assert(not nda::SimdReal<std::complex<float>>);
  static_assert(nda::SimdRealOrComplex<std::complex<double>>);
  static_assert(not nda::SimdRealOrComplex<int>);
  static_assert(nda::SimdComplex<std::complex<float>>);
  static_assert(not nda::SimdComplex<double>);
  static_assert(nda::SimdPreservesLaneWidth<double, std::complex<double>, double>);
  static_assert(nda::SimdPreservesLaneWidth<float, std::complex<float>>);
  static_assert(nda::SimdPreservesLaneWidth<double const &, std::complex<double> const &>);
  static_assert(not nda::SimdPreservesLaneWidth<double, float>);
  static_assert(not nda::SimdPreservesLaneWidth<bool, double>);
}

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
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::reciprocal_f, float, float>);
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::pow_f, float, float>);
  static_assert(not nda::LoadWithNativeSimdTo<nda::detail::abs2_f, float, float>);

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
    nda::native_simd<float> load(nda::native_simd<double> const &) const { return nda::native_simd<float>(0.f); }
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

namespace {
  template <typename T>
  void check_direct_loads() {
    using batch_t = nda::native_simd<T>;
    T value;
    if constexpr (nda::is_complex_v<T>) {
      value = T(0.5, 0.125);
    } else {
      value = T(0.5);
    }
    batch_t batch(value);

    expect_close(nda::detail::conj_f{}.load(batch).get(0), nda::detail::conj(value));
    expect_close(nda::detail::real_f{}.load(batch).get(0), nda::detail::real(value));
    expect_close(nda::detail::imag_f{}.load(batch).get(0), nda::detail::imag(value));

    if constexpr (nda::is_complex_v<T>) { expect_close(nda::detail::abs_f{}.load(batch).get(0), std::abs(value)); }
    if constexpr (std::same_as<nda::remove_complex_t<T>, double>) {
      expect_close(nda::detail::abs2_f{}.load(batch).get(0), nda::detail::abs2(value));
      expect_close(nda::detail::reciprocal_f{}.load(batch).get(0), T(1.0) / value);
      expect_close(nda::detail::pow_f{2.0}.load(batch).get(0), T(std::pow(value, 2.0)));
    }
    if constexpr (not nda::is_complex_v<T>) {
      expect_close(nda::detail::max_f{}.load(batch, batch_t(T(1))).get(0), T(1));
      expect_close(nda::detail::min_f{}.load(batch, batch_t(T(1))).get(0), value);
      expect_close(nda::detail::floor_f{}.load(batch).get(0), std::floor(value));
      expect_close(nda::detail::abs_f{}.load(-batch).get(0), std::abs(-value));
    }
    expect_close(nda::detail::exp_f{}.load(batch).get(0), std::exp(value));
  }

} // namespace

TEST(SIMD, DirectLoad) {
  check_direct_loads<float>();
  check_direct_loads<double>();
  check_direct_loads<std::complex<float>>();
  check_direct_loads<std::complex<double>>();
}
#endif
