// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./simd_test_common.hpp"

#include <nda/simd/mock_simd.hpp>
#include <nda/simd/simd_dispatch.hpp>

#include <vector>

using namespace nda;

namespace {

#ifdef NDA_HAVE_XSIMD
  template <bool Loadable, bool Enabled, typename E>
  void check_dispatch(E const &) {
    static_assert(simd::has_load_function<E>() == Loadable);
    static_assert(is_simd_enabled_v<E> == Enabled);
  }

  template <typename T, typename Layout1, typename Layout2>
  void check_traits() {
    array<T, 2, Layout1> a(32, 32);
    array<T, 2, Layout2> b(32, 32);
    constexpr bool same_layout = std::same_as<Layout1, Layout2>;

    check_dispatch<true, true>(a);
    check_dispatch<true, true>(b);
    check_dispatch<true, same_layout>(a + b);
    check_dispatch<true, same_layout>(a - b);
    check_dispatch<true, same_layout>(a * b);
    check_dispatch<true, same_layout>(a / b);
    check_dispatch<true, same_layout>((a + b) * (a - b));
    check_dispatch<true, same_layout>(-(-(a / b) + T{3}) * T{2});
    check_dispatch<true, true>(a * T{2});
    check_dispatch<true, true>(T{2} * a);
    check_dispatch<true, true>(-a);

    // Mixed array value types cannot be loaded as the result's native batch.
    array<std::int16_t, 2, Layout1> narrow(32, 32);
    check_dispatch<false, false>(a + narrow);
    static_assert(not simd::has_load_function<decltype(a), std::int16_t>());

    auto slice = a(range(0, 16), 0);
    check_dispatch<true, false>(slice);
    check_dispatch<true, false>(slice + array<T, 1>(16));

    struct scalar_f {
      T operator()(T x, T y) const { return x + y; }
    };
    struct mock_f : simd::mock_simd<mock_f, T> {
      T operator()(T x, T y) const { return x + y; }
    };
    struct native_f : scalar_f {
      [[nodiscard]] simd::native_simd<T> load(simd::native_simd<T> x, simd::native_simd<T> y) const { return x + y; }
    };
    struct scalar_load_f : scalar_f {
      [[nodiscard]] T load(T x, T y) const { return x + y; }
    };
    struct wrong_args_f : scalar_f {
      [[nodiscard]] simd::native_simd<T> load(T x, T y) const { return simd::native_simd<T>{x + y}; }
    };
    struct wrong_result_f : scalar_f {
      [[nodiscard]] T load(simd::native_simd<T> x, simd::native_simd<T> y) const { return (x + y).get(0); }
    };

    // Each signature is checked alone, inside arithmetic, and with expression arguments.
    auto check_map = [&]<typename F, bool Loadable>() {
      auto mapped = map(F{})(a, b);
      check_dispatch<Loadable, Loadable and same_layout>(mapped);
      check_dispatch<Loadable, Loadable and same_layout>(mapped + a * b);
      check_dispatch<Loadable, Loadable and same_layout>(map(F{})(a + b, a * b));
    };
    check_map.template operator()<scalar_f, false>();
    check_map.template operator()<mock_f, true>();
    check_map.template operator()<native_f, true>();
    check_map.template operator()<scalar_load_f, false>();
    check_map.template operator()<wrong_args_f, false>();
    check_map.template operator()<wrong_result_f, false>();

    check_dispatch<true, same_layout>(map(native_f{})(a, b) + a * b - T{3} + a + b);
    check_dispatch<false, false>(map(native_f{})(a, b) + map(scalar_f{})(a, b) + a * b);
    check_dispatch<false, false>(map(native_f{})(map(scalar_f{})(a, b), a));
    check_dispatch<true, same_layout>(map(mock_f{})(a, b) * T{2} - T{1});
    check_dispatch<true, false>(map(native_f{})(a, b)(range(0, 16), 0));
  }
#endif

  // Assigns an expression to a rank-2 result and compares every element with expected(i, j).
  template <typename R>
  void check_assignment(R &result, auto const &expression, auto expected) {
    result = expression;
    for (long i = 0; i < result.extent(0); ++i) {
      for (long j = 0; j < result.extent(1); ++j) { expect_simd_eq(result(i, j), get_value_t<R>(expected(i, j))); }
    }
  }

  template <typename T, typename Layout>
  void check_flat_loop() {
    using matrix_t   = matrix<T, Layout>;
    constexpr long n = 2 * simd_width<T>() + 3;
    matrix_t m(n, n), other(n, n), result(n, n);
    fill_random(m, 1, -1000, 1000);
    fill_random(other, 2, -1000, 1000);
    const T scalar{3};

    static_assert(not supports_flat_loop_v<T>);
    static_assert(supports_flat_loop_v<decltype(m(range::all, 0))>);
    static_assert(supports_flat_loop_v<matrix_t>);
    static_assert(supports_flat_loop_v<decltype(m + other)>);
    static_assert(supports_flat_loop_v<decltype(scalar * m - other)>);
    static_assert(supports_flat_loop_v<decltype(-(m + other))>);
    static_assert(supports_flat_loop_v<decltype(abs(m + other))>);
    static_assert(not supports_flat_loop_v<decltype(m + scalar)>);
    static_assert(not supports_flat_loop_v<decltype(scalar - m)>);
    static_assert(not supports_flat_loop_v<decltype(m + (m + (m + scalar) + scalar))>);
    static_assert(not supports_flat_loop_v<decltype(scalar * (m - scalar))>);
    static_assert(not supports_flat_loop_v<decltype(-(m + scalar))>);
    static_assert(not supports_flat_loop_v<decltype(abs(m + scalar))>);

    // matrix +/- scalar reports the layout of the matrix, like every other expression with a scalar operand
    static_assert(has_contiguous_layout<decltype(m + scalar)>);
    static_assert(get_layout_info<decltype(scalar - m)>.stride_order == get_layout_info<matrix_t>.stride_order);
    static_assert(has_contiguous_layout<decltype(abs(m))>);

    // matrix +/- scalar must still take the SIMD path (diagonal lane select), not the scalar fallback
    static_assert(is_simd_enabled_v<decltype(m + scalar)> == Vectorizable<T>);
    static_assert(is_simd_enabled_v<decltype(scalar - m)> == Vectorizable<T>);

    auto diagonal = [&](long i, long j) { return i == j ? scalar : T{0}; };
    check_assignment(result, m + other, [&](long i, long j) { return m(i, j) + other(i, j); });
    check_assignment(result, scalar * m - other, [&](long i, long j) { return scalar * m(i, j) - other(i, j); });
    check_assignment(result, m + scalar, [&](long i, long j) { return m(i, j) + diagonal(i, j); });
    check_assignment(result, scalar - m, [&](long i, long j) { return diagonal(i, j) - m(i, j); });
    check_assignment(result, m + (m + (m + scalar) + scalar), [&](long i, long j) { return T{3} * m(i, j) + T{2} * diagonal(i, j); });

    // Rectangular matrices only add/subtract the scalar on the shorter diagonal.
    matrix_t rectangular(n, n + 5), out(n, n + 5);
    fill_random(rectangular, 3, -1000, 1000);
    check_assignment(out, rectangular - scalar, [&](long i, long j) { return rectangular(i, j) - diagonal(i, j); });
  }

} // namespace

template <typename T>
class SIMDDispatch : public ::testing::Test {};
TYPED_TEST_SUITE(SIMDDispatch, simd_types);

// The expression tests use abs, which has no overload for unsigned types.
template <typename T>
class SIMDExpressions : public ::testing::Test {};
using signed_simd_types = ::testing::Types<std::int32_t, std::int64_t, float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SIMDExpressions, signed_simd_types);

#ifdef NDA_HAVE_XSIMD
TYPED_TEST(SIMDDispatch, Traits) {
  check_traits<TypeParam, C_layout, C_layout>();
  check_traits<TypeParam, C_layout, F_layout>();
  check_traits<TypeParam, F_layout, C_layout>();
  check_traits<TypeParam, F_layout, F_layout>();
}
#endif

TEST(SIMD, UnsupportedTypes) {
  static_assert(not simd::has_load_function<array<std::vector<int>, 2>>());
  static_assert(not is_simd_enabled_v<array<std::vector<int>, 2>>);
  static_assert(not simd::has_load_function<array<float *, 3>>());
  static_assert(not is_simd_enabled_v<array<float *, 3>>);
  static_assert(not is_simd_enabled_v<float>);
#ifndef NDA_HAVE_XSIMD
  static_assert(not is_simd_enabled_v<array<double, 1>>);
  static_assert(not is_simd_enabled_v<decltype(abs(std::declval<array<std::complex<double>, 1> &>()))>);
#endif
}

TEST(SIMD, ScalarBroadcast) {
  array<float, 2> a(5, 7);
  a = 1.5f;
#ifdef NDA_HAVE_XSIMD
  static_assert(is_simd_enabled_v<decltype(a * 2)>);
  static_assert(is_simd_enabled_v<decltype(2 + a)>);
  static_assert(is_simd_enabled_v<decltype(a - 1)>);
  static_assert(not is_simd_enabled_v<decltype(a + 1.0)>);
#endif
  array<float, 2> result = a * 2 - 1;
  for (auto x : result) { EXPECT_EQ(x, 2.0f); }
  matrix<float> m(a), out(5, 7);
  check_assignment(out, m + 1, [](long i, long j) { return i == j ? 2.5f : 1.5f; });
}

TEST(SIMD, ScalarTree) {
  array<float, 2> a(5, 7), b(5, 7), c(5, 7);
  auto native = a + b + c;
  auto scalar = map([](auto const &x) { return x * x; })(native + a);
  auto negated = -scalar;
  auto scaled = 5.0f * (negated + 5.0f);
  auto mapped = map([](auto const &x, auto const &y) { return x - y; })(scaled + scaled, scalar);
#ifdef NDA_HAVE_XSIMD
  static_assert(is_simd_enabled_v<decltype(native)>);
#else
  static_assert(not is_simd_enabled_v<decltype(native)>);
#endif
  static_assert(not is_simd_enabled_v<decltype(scalar)>);
  static_assert(not is_simd_enabled_v<decltype(negated)>);
  static_assert(not is_simd_enabled_v<decltype(scaled)>);
  static_assert(not is_simd_enabled_v<decltype(mapped)>);
  static_assert(not is_simd_enabled_v<decltype(log(mapped))>);
}

TYPED_TEST(SIMDExpressions, FlatLoop) {
  check_flat_loop<TypeParam, C_layout>();
  check_flat_loop<TypeParam, F_layout>();
}

#ifdef NDA_HAVE_XSIMD
// for_each_simd falls back to a scalar loop over multi-indices if the arrays cannot be looped over in SIMD-width
// blocks, without instantiating f_simd (here it would not compile).
TEST(SIMD, ForEachSimdScalarFallback) {
  constexpr long n = 2 * simd::native_simd<double>::size + 3;
  auto no_simd     = []<typename... I>(I const &...) { static_assert(always_false<I...>, "f_simd must not be instantiated"); };
  auto check       = [&](auto const &first, auto const &...rest) {
    using idx_t = std::array<long, get_rank<decltype(first)>>;
    std::vector<idx_t> visited, expected;
    // the callable receives the arrays themselves (not copies), followed by the multi-index
    if constexpr (sizeof...(rest) == 0) {
      for_each_simd(no_simd, [&](auto const &x, auto const &...i) { EXPECT_EQ(&x, &first); visited.push_back({i...}); }, first);
    } else {
      for_each_simd(no_simd, [&](auto const &x, auto const &, auto const &...i) { EXPECT_EQ(&x, &first); visited.push_back({i...}); }, first,
                    rest...);
    }
    nda::for_each(first.shape(), [&](auto const &...i) { expected.push_back({i...}); });
    EXPECT_EQ(visited, expected);
  };

  array<double, 2> c(n, n);
  array<double, 2, F_layout> f(n, n);
  array<double, 1> d(n);
  array<std::int64_t, 1> l(n);
  array<long double, 1> ld(n);
  check(c(range::all, range(0, n, 2))); // strided fastest dimension
  check(c, f);                          // different stride orders
  check(d, l);                          // different value types with the same SIMD width
  static_assert(simd::native_simd<double>::size == simd::native_simd<std::int64_t>::size);
  check(ld); // no native batch
  static_assert(not Vectorizable<long double>);
  check(map([](double x) { return x * x; })(d)); // functor without a load function
}

TEST(SIMD, ForEachSimdBatchesAndTail) {
  constexpr long lanes = simd::native_simd<double>::size;
  constexpr long n     = 2 * lanes + 3;
  array<double, 1> d(n);
  long blocks = 0, elements = 0;
  for_each_simd([&](auto const &...) { ++blocks; }, [&](auto const &...) { ++elements; }, d, d + d);
  EXPECT_EQ(blocks, n / lanes);
  EXPECT_EQ(elements, n % lanes);
}
#endif
