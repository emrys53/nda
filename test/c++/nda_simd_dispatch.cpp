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
    check_dispatch<true, same_layout>((a - b) / (a / b) + a * b);
    check_dispatch<true, same_layout>(a * b + a * b * (a + b));
    check_dispatch<true, same_layout>(-(-(a / b) + T{3}) * T{2});
    check_dispatch<true, true>(a * T{2});
    check_dispatch<true, true>(T{2} * a);
    check_dispatch<true, true>(-a);

    // Mixed array value types cannot be loaded as the result's native batch.
    array<std::int16_t, 2, Layout1> narrow(32, 32);
    check_dispatch<false, false>(a + narrow);
    check_dispatch<false, false>((a + narrow) * (a + b));
    check_dispatch<false, false>(a + b + (a + narrow) - narrow);
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
      native_simd<T> load(native_simd<T> x, native_simd<T> y) const { return x + y; }
    };
    struct scalar_load_f : scalar_f {
      T load(T x, T y) const { return x + y; }
    };
    struct wrong_args_f : scalar_f {
      native_simd<T> load(T x, T y) const { return native_simd<T>{x + y}; }
    };
    struct wrong_result_f : scalar_f {
      T load(native_simd<T> x, native_simd<T> y) const { return (x + y).get(0); }
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

  template <typename T, typename Layout>
  void check_flat_loop() {
    using matrix_t = matrix<T, Layout>;
    constexpr long n = [] {
#ifdef NDA_HAVE_XSIMD
      if constexpr (Vectorizable<T>) {
        return long(2 * native_simd<T>::size + 3);
      }
#endif
      return 19L;
    }();
    matrix_t m(n, n), other(n, n), result(n, n);
    for (long i = 0; i < n; ++i) {
      for (long j = 0; j < n; ++j) {
        m(i, j) = T(1 + i + 10 * j);
        other(i, j) = T(2 * i - j);
      }
    }
    const T scalar{3};

    static_assert(std::is_base_of_v<std::false_type, supports_flat_loop<T>>);
    static_assert(std::is_base_of_v<std::true_type, supports_flat_loop<matrix_t>>);
    static_assert(not supports_flat_loop_v<T>);
    static_assert(not supports_flat_loop_v<int>);
    static_assert(supports_flat_loop_v<decltype(m(range::all, 0))>);
    static_assert(supports_flat_loop_v<matrix_t>);
    static_assert(supports_flat_loop_v<decltype(m + other)>);
    static_assert(supports_flat_loop_v<decltype(scalar * m - other)>);
    static_assert(supports_flat_loop_v<decltype(-(m + other))>);
    static_assert(supports_flat_loop_v<decltype(abs(m + other))>);
    static_assert(supports_flat_loop_v<decltype(scalar * m)>);
    static_assert(not supports_flat_loop_v<decltype(m + scalar)>);
    static_assert(not supports_flat_loop_v<decltype(scalar - m)>);
    static_assert(not supports_flat_loop_v<decltype(m + (m + (m + scalar) + scalar))>);
    static_assert(not supports_flat_loop_v<decltype(scalar * (m - scalar))>);
    static_assert(not supports_flat_loop_v<decltype(-(m + scalar))>);
    static_assert(not supports_flat_loop_v<decltype(abs(m + scalar))>);

    auto check = [&](auto const &expression, auto expected) {
      result = expression;
      for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) { expect_simd_eq(result(i, j), T(expected(i, j))); }
      }
    };
    check(m + other, [&](long i, long j) { return m(i, j) + other(i, j); });
    check(scalar * m - other, [&](long i, long j) { return scalar * m(i, j) - other(i, j); });
    check(m + scalar, [&](long i, long j) { return m(i, j) + (i == j ? scalar : T{0}); });
    check(scalar - m, [&](long i, long j) { return (i == j ? scalar : T{0}) - m(i, j); });
    check(m + (m + (m + scalar) + scalar), [&](long i, long j) { return T{3} * m(i, j) + (i == j ? T{2} * scalar : T{0}); });

    // Rectangular matrices only add/subtract the scalar on the shorter diagonal.
    matrix_t rectangular(n, n + 5), out(n, n + 5);
    for (long i = 0; i < n; ++i) {
      for (long j = 0; j < n + 5; ++j) { rectangular(i, j) = T(i - j); }
    }
    out = rectangular - scalar;
    for (long i = 0; i < n; ++i) {
      for (long j = 0; j < n + 5; ++j) { expect_simd_eq(out(i, j), T(rectangular(i, j) - (i == j ? scalar : T{0}))); }
    }
  }

} // namespace

#ifdef NDA_HAVE_XSIMD
template <typename T>
class SIMDDispatch : public ::testing::Test {};

TYPED_TEST_SUITE(SIMDDispatch, simd_types);

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
  for (long i = 0; i < 5; ++i) {
    for (long j = 0; j < 7; ++j) { EXPECT_EQ(result(i, j), a(i, j) * 2 - 1); }
  }
  matrix<float> m(5, 7), out(5, 7);
  for (long i = 0; i < 5; ++i) {
    for (long j = 0; j < 7; ++j) { m(i, j) = 1.5f; }
  }
  out = m + 1;
  for (long i = 0; i < 5; ++i) {
    for (long j = 0; j < 7; ++j) { EXPECT_EQ(out(i, j), i == j ? 2.5f : 1.5f); }
  }
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

TEST(SIMD, FlatLoop) {
  check_flat_loop<float, C_layout>();
  check_flat_loop<float, F_layout>();
  check_flat_loop<double, C_layout>();
  check_flat_loop<double, F_layout>();
  check_flat_loop<std::int32_t, C_layout>();
  check_flat_loop<std::int64_t, F_layout>();
  check_flat_loop<std::complex<double>, C_layout>();
  check_flat_loop<std::complex<double>, F_layout>();
}

#ifdef NDA_HAVE_XSIMD
namespace {

  template <typename T, typename Layout>
  void check_row_contiguous_views() {
    constexpr long lanes = native_simd<T>::size;
    for (long n : {lanes - 1, 2 * lanes + 3}) {
      array<T, 3, Layout> t(n, 3, n), other(n, 3, n), full(n, 3, n);
      for (long i = 0; i < n; ++i) {
        for (long j = 0; j < 3; ++j) {
          for (long k = 0; k < n; ++k) {
            t(i, j, k)     = T(1 + i + 10 * k);
            other(i, j, k) = T(2 * i - k);
            full(i, j, k)  = T(-1);
          }
        }
      }
      auto tv = t(range::all, 1, range::all);
      auto ov = other(range::all, 1, range::all);
      auto fv = full(range::all, 1, range::all);
      array<T, 2, Layout> c(n, n), result(n, n);
      c = T(3);

      static_assert(not has_contiguous_layout<decltype(tv)>);
      static_assert(has_layout_smallest_stride_is_one<decltype(tv)>);
      static_assert(supports_flat_loop_v<decltype(tv)>);
      check_dispatch<true, true>(tv);
      check_dispatch<true, true>(tv * ov + tv);
      check_dispatch<true, true>(tv + c);
      check_dispatch<true, true>(-(tv - ov) * T{2});
      check_dispatch<true, true>(abs(tv));
      if constexpr (array<T, 3, Layout>::layout_t::is_stride_order_C()) {
        check_dispatch<true, false>(t(range::all, range::all, 1));
      } else {
        check_dispatch<true, false>(t(1, range::all, range::all));
      }

      auto check = [&](auto const &expression, auto expected) {
        fv = expression;
        for (long i = 0; i < n; ++i) {
          for (long k = 0; k < n; ++k) {
            expect_simd_eq(full(i, 1, k), T(expected(i, k)));
            expect_simd_eq(full(i, 0, k), T(-1));
            expect_simd_eq(full(i, 2, k), T(-1));
          }
        }
      };
      check(tv * ov + tv, [&](long i, long k) { return t(i, 1, k) * other(i, 1, k) + t(i, 1, k); });
      check(tv + c, [&](long i, long k) { return t(i, 1, k) + T(3); });
      check(-(tv - ov) * T{2}, [&](long i, long k) { return -(t(i, 1, k) - other(i, 1, k)) * T{2}; });

      result = tv - ov;
      T total{};
      for (long i = 0; i < n; ++i) {
        for (long k = 0; k < n; ++k) {
          expect_simd_eq(result(i, k), T(t(i, 1, k) - other(i, 1, k)));
          total += t(i, 1, k);
        }
      }
      expect_simd_eq(sum(tv), total);
      if constexpr (not is_complex_v<T>) {
        expect_simd_eq(max_element(tv), T(1 + (n - 1) + 10 * (n - 1)));
        expect_simd_eq(min_element(tv), T(1));
      }
    }
  }

  template <typename T>
  void check_row_contiguous_views_mixed_stride_order() {
    constexpr long n = 2 * native_simd<T>::size + 3;
    array<T, 3> c(n, 3, n);
    array<T, 3, F_layout> f(n, 3, n);
    for (long i = 0; i < n; ++i) {
      for (long j = 0; j < 3; ++j) {
        for (long k = 0; k < n; ++k) {
          c(i, j, k) = T(1 + i + 10 * k);
          f(i, j, k) = T(2 * i - k);
        }
      }
    }
    auto cv  = c(range::all, 1, range::all);
    auto fv  = f(range::all, 1, range::all);
    auto cvt = transpose(cv);
    array<T, 2> rc(n, n);
    array<T, 2, F_layout> rf(n, n);

    static_assert(has_layout_smallest_stride_is_one<decltype(fv)> and has_layout_smallest_stride_is_one<decltype(cvt)>);
    static_assert(get_layout_info<decltype(cv)>.stride_order != get_layout_info<decltype(fv)>.stride_order);
    check_dispatch<true, true>(cv + cv);
    check_dispatch<true, true>(fv + fv);
    check_dispatch<true, true>(cvt + cvt);
    check_dispatch<true, false>(cv + fv);
    check_dispatch<true, false>(cv + cvt);
    check_dispatch<true, false>(cv * fv - cvt);

    auto check = [&](auto &result, auto const &expression, auto expected) {
      result = expression;
      for (long i = 0; i < n; ++i) {
        for (long k = 0; k < n; ++k) { expect_simd_eq(result(i, k), T(expected(i, k))); }
      }
    };
    check(rc, cv + fv, [&](long i, long k) { return c(i, 1, k) + f(i, 1, k); });
    check(rf, cv + fv, [&](long i, long k) { return c(i, 1, k) + f(i, 1, k); });
    check(rf, cv + cv, [&](long i, long k) { return T{2} * c(i, 1, k); });
    check(rc, cv + cvt, [&](long i, long k) { return c(i, 1, k) + c(k, 1, i); });
    check(rf, cvt * fv, [&](long i, long k) { return c(k, 1, i) * f(i, 1, k); });
  }

} // namespace

TEST(SIMD, RowContiguousViews) {
  check_row_contiguous_views<float, C_layout>();
  check_row_contiguous_views<float, F_layout>();
  check_row_contiguous_views<double, C_layout>();
  check_row_contiguous_views<double, F_layout>();
  check_row_contiguous_views<std::int32_t, C_layout>();
  check_row_contiguous_views<std::int64_t, F_layout>();
  check_row_contiguous_views<std::complex<double>, C_layout>();
  check_row_contiguous_views<std::complex<double>, F_layout>();
}

TEST(SIMD, RowContiguousViewsMixedStrideOrder) {
  check_row_contiguous_views_mixed_stride_order<float>();
  check_row_contiguous_views_mixed_stride_order<double>();
  check_row_contiguous_views_mixed_stride_order<std::int64_t>();
  check_row_contiguous_views_mixed_stride_order<std::complex<double>>();
}
#endif
