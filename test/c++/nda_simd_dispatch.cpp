// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./simd_test_common.hpp"

#include <nda/simd/mock_simd.hpp>
#include <nda/simd/simd_dispatch.hpp>

#include <algorithm>
#include <vector>

using namespace nda;

namespace {

  template <bool Loadable, bool Enabled, typename E>
  void check_dispatch(E const &) {
    static_assert(has_simd_load_v<E> == Loadable);
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
    static_assert(not has_simd_load_v<decltype(a), std::int16_t>);

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

  // Assigns an expression to a rank-2 result and compares every element with expected(i, j).
  template <typename R>
  void check_assignment(R &result, auto const &expression, auto expected) {
    result = expression;
    for (long i = 0; i < result.extent(0); ++i) {
      for (long j = 0; j < result.extent(1); ++j) { expect_simd_eq(result(i, j), get_value_t<R>(expected(i, j))); }
    }
  }

  template <typename T, typename Layout>
  void check_elementwise() {
    using matrix_t   = matrix<T, Layout>;
    constexpr long n = 2 * simd::native_simd<T>::size + 3;
    matrix_t m(n, n), other(n, n), result(n, n);
    fill_random(m, 1, -1000, 1000);
    fill_random(other, 2, -1000, 1000);
    const T scalar{3};

    static_assert(not is_elementwise_v<T>);
    static_assert(is_elementwise_v<decltype(m(range::all, 0))>);
    static_assert(is_elementwise_v<matrix_t>);
    static_assert(is_elementwise_v<decltype(m + other)>);
    static_assert(is_elementwise_v<decltype(scalar * m - other)>);
    static_assert(is_elementwise_v<decltype(-(m + other))>);
    static_assert(is_elementwise_v<decltype(abs(m + other))>);
    static_assert(not is_elementwise_v<decltype(m + scalar)>);
    static_assert(not is_elementwise_v<decltype(scalar - m)>);
    static_assert(not is_elementwise_v<decltype(m + (m + (m + scalar) + scalar))>);
    static_assert(not is_elementwise_v<decltype(scalar * (m - scalar))>);
    static_assert(not is_elementwise_v<decltype(-(m + scalar))>);
    static_assert(not is_elementwise_v<decltype(abs(m + scalar))>);

    // matrix +/- scalar reports the layout of the matrix, like every other expression with a scalar operand
    static_assert(has_contiguous_layout<decltype(m + scalar)>);
    static_assert(get_layout_info<decltype(scalar - m)>.stride_order == get_layout_info<matrix_t>.stride_order);
    static_assert(has_contiguous_layout<decltype(abs(m))>);

    // matrix +/- scalar must still take the SIMD path (diagonal lane select), not the scalar fallback
    static_assert(is_simd_enabled_v<decltype(m + scalar)> == simd::Vectorizable<T>);
    static_assert(is_simd_enabled_v<decltype(scalar - m)> == simd::Vectorizable<T>);

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

// The expression tests fill arrays with negative values, so they use signed types only.
template <typename T>
class SIMDExpressions : public ::testing::Test {};
using signed_simd_types = ::testing::Types<std::int32_t, std::int64_t, float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SIMDExpressions, signed_simd_types);

TYPED_TEST(SIMDDispatch, Traits) {
  check_traits<TypeParam, C_layout, C_layout>();
  check_traits<TypeParam, C_layout, F_layout>();
  check_traits<TypeParam, F_layout, C_layout>();
  check_traits<TypeParam, F_layout, F_layout>();
}

TEST(SIMD, UnsupportedTypes) {
  static_assert(not has_simd_load_v<array<std::vector<int>, 2>>);
  static_assert(not is_simd_enabled_v<array<std::vector<int>, 2>>);
  static_assert(not has_simd_load_v<array<float *, 3>>);
  static_assert(not is_simd_enabled_v<array<float *, 3>>);
  static_assert(not is_simd_enabled_v<float>);
}

// A mock functor loads any native batches: the scalar call converts the lanes. Batches with a different lane count
// than the functor's result keep the expression scalar.
TEST(SIMD, MockFunctorArguments) {
  struct plus_one : simd::mock_simd<plus_one, double> {
    double operator()(double x) const { return x + 1; }
  };
  array<std::int64_t, 1> l(2 * simd::native_simd<double>::size + 3);
  l = 1;
  static_assert(is_simd_enabled_v<decltype(map(plus_one{})(l))>);
  array<double, 1> const d = map(plus_one{})(l);
  for (auto x : d) { EXPECT_EQ(x, 2.0); }
  array<std::int32_t, 1> s(l.size());
  static_assert(not is_simd_enabled_v<decltype(map(plus_one{})(s))>);
}

TEST(SIMD, ScalarBroadcast) {
  array<float, 2> a(5, 7);
  a = 1.5f;
  static_assert(is_simd_enabled_v<decltype(a * 2)>);
  static_assert(is_simd_enabled_v<decltype(2 + a)>);
  static_assert(is_simd_enabled_v<decltype(a - 1)>);
  static_assert(not is_simd_enabled_v<decltype(a + 1.0)>);
  array<float, 2> result = a * 2 - 1;
  for (auto x : result) { EXPECT_EQ(x, 2.0f); }
  matrix<float> m(a), out(5, 7);
  check_assignment(out, m + 1, [](long i, long j) { return i == j ? 2.5f : 1.5f; });
  // a scalar type without a batch of its own is converted to the value type before it is broadcast
  static_assert(is_simd_enabled_v<decltype(m + true)>);
  check_assignment(out, m + true, [](long i, long j) { return i == j ? 2.5f : 1.5f; });
}

// A real scalar combined with a complex array acts on the components, like std::complex<T> op T: the batches agree
// with the scalar path bit for bit, also for magnitudes and infinities where a complex division or product would not.
TEST(SIMD, RealScalarOnComplex) {
  using C          = std::complex<double>;
  constexpr long n = 4 * simd::native_simd<C>::size + 3;
  array<C, 1> z(n);
  auto expect_same = [&](auto const &expression, auto reference) {
    static_assert(is_simd_enabled_v<decltype(expression)>);
    array<C, 1> const result = expression;
    for (long i = 0; i < n; ++i) { EXPECT_EQ(result(i), reference(z(i))); }
  };
  fill_random(z, 7, -10, 10);
  expect_same(z + 1.5, [](C x) { return x + 1.5; });
  expect_same(1.5 + z, [](C x) { return 1.5 + x; });
  expect_same(z - 1.5, [](C x) { return x - 1.5; });
  expect_same(1.5 - z, [](C x) { return 1.5 - x; });
  expect_same(z * 3.0, [](C x) { return x * 3.0; });
  expect_same(3.0 * z, [](C x) { return 3.0 * x; });
  expect_same(z / 3.0, [](C x) { return x / 3.0; });
  double const inf = std::numeric_limits<double>::infinity();
  z                = C{1, 1};
  expect_same(z / 1e200, [](C x) { return x / 1e200; }); // (1e-200, 1e-200), not (0, 0)
  z = C{inf, 1};
  expect_same(z / 2.0, [](C x) { return x / 2.0; }); // (inf, 0.5), not (inf, NaN)
  z = C{1, inf};
  expect_same(2.0 * z, [](C x) { return 2.0 * x; }); // (2, inf), not (NaN, inf)

  // the same holds for the diagonal lane of matrix +/- scalar
  matrix<C> m(n, n);
  fill_random(m, 8, -10, 10);
  matrix<C, F_layout> const f(m);
  matrix<C> const r1           = m - 2.0;
  matrix<C, F_layout> const r2 = 2.0 - f;
  for (long i = 0; i < n; ++i) {
    for (long j = 0; j < n; ++j) {
      EXPECT_EQ(r1(i, j), (i == j ? m(i, j) - 2.0 : m(i, j)));
      EXPECT_EQ(r2(i, j), (i == j ? 2.0 - f(i, j) : -f(i, j)));
    }
  }
}

TEST(SIMD, ScalarTree) {
  array<float, 2> a(5, 7), b(5, 7), c(5, 7);
  auto native  = a + b + c;
  auto scalar  = map([](auto const &x) { return x * x; })(native + a);
  auto negated = -scalar;
  auto scaled  = 5.0f * (negated + 5.0f);
  auto mapped  = map([](auto const &x, auto const &y) { return x - y; })(scaled + scaled, scalar);
  static_assert(is_simd_enabled_v<decltype(native)>);
  static_assert(not is_simd_enabled_v<decltype(scalar)>);
  static_assert(not is_simd_enabled_v<decltype(negated)>);
  static_assert(not is_simd_enabled_v<decltype(scaled)>);
  static_assert(not is_simd_enabled_v<decltype(mapped)>);
  static_assert(not is_simd_enabled_v<decltype(log(mapped))>);
}

// Reductions at sizes around the three loops of nda::detail::simd_reduce (four accumulators, single batches, scalar
// tail): each boundary from both sides, and all three loops in one run (9w + 3). frobenius_norm also runs on a
// row-contiguous view.
TYPED_TEST(SIMDDispatch, Reductions) {
  using T          = TypeParam;
  constexpr long w = simd::native_simd<T>::size;
  for (long n : {1L, w - 1, w, w + 1, 4 * w - 1, 4 * w, 4 * w + 1, 9 * w + 3}) {
    array<T, 1> a(n);
    fill_random(a, static_cast<unsigned>(n), 0, 1000);
    T total{};
    for (auto x : a) { total += x; }
    expect_simd_eq(sum(a), total);
    // init is folded in once, the identity fills the accumulator lanes
    auto add = [](auto const &x) { return xsimd::reduce_add(x); };
    expect_simd_eq(nda::detail::simd_reduce(a, std::plus<>{}, add, T{0}, T{5}), T{5} + total);

    // a single distinguished element in each of the four accumulators, in the single-batch loop and in the tail catches
    // a dropped or repeated lane anywhere in the loop
    for (long p : {0L, w - 1, w, 2 * w, 3 * w, 4 * w - 1, 4 * w, 8 * w, n - 1}) {
      if (p >= n) continue;
      a    = T{1};
      a(p) = T{3};
      expect_simd_eq(product(a), T{3});
      expect_simd_eq(nda::detail::simd_reduce(a, std::multiplies<>{}, [](auto const &x) { return xsimd::reduce_mul(x); }, T{1}, T{2}), T{6});
      if constexpr (not is_complex_v<T>) {
        expect_simd_eq(max_element(a), T{3});
        a(p) = T{0};
        expect_simd_eq(min_element(a), T{0});
      }
    }

    // single digits keep the sum of squares exact in float accumulators
    array<T, 2> m(2, n + 3);
    fill_random(m, static_cast<unsigned>(n), 0, 10);
    auto frobenius = [](auto const &x) {
      double r = 0;
      for (auto v : x) { r += static_cast<double>(std::norm(v)); }
      return std::sqrt(r);
    };
    auto view    = m(range::all, range(1, n + 1));
    using norm_t = decltype(frobenius_norm(m));
    expect_simd_eq(frobenius_norm(m), static_cast<norm_t>(frobenius(m)));
    expect_simd_eq(frobenius_norm(view), static_cast<norm_t>(frobenius(view)));
    T view_total{};
    for (auto x : view) { view_total += x; }
    expect_simd_eq(nda::detail::simd_reduce(view, std::plus<>{}, add, T{0}, T{5}), T{5} + view_total);
  }
}

TYPED_TEST(SIMDExpressions, Elementwise) {
  check_elementwise<TypeParam, C_layout>();
  check_elementwise<TypeParam, F_layout>();
}

namespace {

  template <typename T, typename Layout>
  void check_row_contiguous_views() {
    constexpr long lanes = simd::native_simd<T>::size;
    for (long n : {lanes - 1, 2 * lanes + 3}) {
      array<T, 3, Layout> t(n, 3, n), other(n, 3, n), full(n, 3, n);
      fill_random(t, 1, -1000, 1000);
      fill_random(other, 2, -1000, 1000);
      full    = T(-1);
      auto tv = t(range::all, 1, range::all);
      auto ov = other(range::all, 1, range::all);
      auto fv = full(range::all, 1, range::all);
      array<T, 2, Layout> c(n, n), result(n, n);
      c = T(3);

      static_assert(not has_contiguous_layout<decltype(tv)>);
      static_assert(has_layout_smallest_stride_is_one<decltype(tv)>);
      static_assert(is_elementwise_v<decltype(tv)>);
      check_dispatch<true, true>(tv);
      check_dispatch<true, true>(tv * ov + tv);
      check_dispatch<true, true>(tv + c);
      check_dispatch<true, true>(-(tv - ov) * T{2});
      check_dispatch<true, true>(abs(tv));
      if constexpr (array<T, 3, Layout>::layout_t::is_stride_order_C()) {
        check_dispatch<true, false>(t(range::all, range::all, 0));
      } else {
        check_dispatch<true, false>(t(0, range::all, range::all));
      }

      check_assignment(fv, tv * ov + tv, [&](long i, long k) { return t(i, 1, k) * other(i, 1, k) + t(i, 1, k); });
      check_assignment(fv, tv + c, [&](long i, long k) { return t(i, 1, k) + T(3); });
      check_assignment(fv, -(tv - ov) * T{2}, [&](long i, long k) { return -(t(i, 1, k) - other(i, 1, k)) * T{2}; });
      check_assignment(result, tv - ov, [&](long i, long k) { return t(i, 1, k) - other(i, 1, k); });
      // the planes next to the assigned one are untouched
      for (auto x : full(range::all, 0, range::all)) { expect_simd_eq(x, T(-1)); }
      for (auto x : full(range::all, 2, range::all)) { expect_simd_eq(x, T(-1)); }

      T total{};
      for (auto x : tv) { total += x; }
      expect_simd_eq(sum(tv), total);
      if constexpr (not is_complex_v<T>) {
        expect_simd_eq(max_element(tv), *std::max_element(tv.begin(), tv.end()));
        expect_simd_eq(min_element(tv), *std::min_element(tv.begin(), tv.end()));
      }
    }
  }

  template <typename T>
  void check_row_contiguous_views_mixed_stride_order() {
    constexpr long n = 2 * simd::native_simd<T>::size + 3;
    array<T, 3> c(n, 3, n);
    array<T, 3, F_layout> f(n, 3, n);
    fill_random(c, 1, -1000, 1000);
    fill_random(f, 2, -1000, 1000);
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

    check_assignment(rc, cv + fv, [&](long i, long k) { return c(i, 1, k) + f(i, 1, k); });
    check_assignment(rf, cv + fv, [&](long i, long k) { return c(i, 1, k) + f(i, 1, k); });
    check_assignment(rf, cv + cv, [&](long i, long k) { return T{2} * c(i, 1, k); });
    check_assignment(rc, cv + cvt, [&](long i, long k) { return c(i, 1, k) + c(k, 1, i); });
    check_assignment(rf, cvt * fv, [&](long i, long k) { return c(k, 1, i) * f(i, 1, k); });
  }

} // namespace

TYPED_TEST(SIMDExpressions, RowContiguousViews) {
  check_row_contiguous_views<TypeParam, C_layout>();
  check_row_contiguous_views<TypeParam, F_layout>();
}

TYPED_TEST(SIMDExpressions, RowContiguousViewsMixedStrideOrder) { check_row_contiguous_views_mixed_stride_order<TypeParam>(); }

// An operand that overlaps the destination other than exactly in place is evaluated element by element, so the
// result does not depend on the batch width; the exact in-place update takes the SIMD path.
TEST(SIMD, OverlappingAssignment) {
  constexpr long w = simd::native_simd<double>::size;
  constexpr long n = 2 * w + 5;
  array<double, 1> buffer(n + 1);
  buffer = 0.0;
  array_contiguous_view<double, 1> src{{n}, buffer.data()}, dst{{n}, buffer.data() + 1}; // dst is src shifted by one
  static_assert(is_simd_enabled_v<decltype(src + 1.0)>);
  EXPECT_EQ(mem::detect_aliasing(dst, src + 1.0), mem::aliasing::may_overlap);
  dst = src + 1.0; // every element sees the element before it, which was just written
  for (long i = 0; i <= n; ++i) { EXPECT_EQ(buffer(i), double(i)); }

  auto whole = buffer();
  EXPECT_EQ(mem::detect_aliasing(whole, whole * 2.0), mem::aliasing::in_place);
  whole = whole * 2.0;
  for (long i = 0; i <= n; ++i) { EXPECT_EQ(buffer(i), 2.0 * double(i)); }
}

TEST(SIMD, ForEachSimdBatchesAndTail) {
  // f_simd is called for every full block and f_scalar for the remaining elements: in one loop over the linear index
  // if all arrays are contiguous and elementwise, otherwise row by row along the fastest dimension, where blocks never
  // cross a row and every row has its own tail
  constexpr long lanes = simd::native_simd<double>::size;
  constexpr long n     = 2 * lanes + 1; // two blocks and one remaining element

  // contiguous and elementwise: one loop over the linear index
  array<double, 1> d(n);
  std::vector<long> blocks, tail;
  for_each_simd([&](auto const &, auto const &, _linear_index_t k) { blocks.push_back(k.value); },
                [&](auto const &, auto const &, _linear_index_t k) { tail.push_back(k.value); }, d, d + d);
  EXPECT_EQ(blocks, (std::vector<long>{0, lanes}));
  EXPECT_EQ(tail, (std::vector<long>{2 * lanes}));

  // otherwise row by row: two blocks and one remaining element in each of the n rows
  auto check_rows = [&](auto const &a, int fastest) {
    std::vector<std::array<long, 2>> row_blocks, row_tail, expected_blocks, expected_tail;
    for_each_simd([&](auto const &, long i, long j) { row_blocks.push_back({i, j}); },
                  [&](auto const &, long i, long j) { row_tail.push_back({i, j}); }, a);
    for (long r = 0; r < n; ++r) {
      auto at = [&](long k) { return fastest == 1 ? std::array{r, k} : std::array{k, r}; };
      expected_blocks.push_back(at(0));
      expected_blocks.push_back(at(lanes));
      expected_tail.push_back(at(2 * lanes));
    }
    EXPECT_EQ(row_blocks, expected_blocks);
    EXPECT_EQ(row_tail, expected_tail);
  };
  // contiguous only along the fastest dimension: the middle plane of rank-3 arrays, in C and F order
  array<double, 3> c3(n, 3, n);
  array<double, 3, F_layout> f3(n, 3, n);
  check_rows(c3(range::all, 1, range::all), 1);
  check_rows(f3(range::all, 1, range::all), 0);
  check_rows(c3(range::all, 1, range::all) * 2.0, 1);
  // contiguous but not elementwise (matrix + scalar acts on the diagonal)
  matrix<double> m(n, n);
  check_rows(m + 1.0, 1);

  // rows shorter than a block are visited by f_scalar only
  array<double, 3> s3(3, 3, lanes - 1);
  auto short_rows = s3(range::all, 1, range::all);
  static_assert(is_simd_enabled_v<decltype(short_rows)>);
  long nblocks = 0, nelements = 0;
  for_each_simd([&](auto const &...) { ++nblocks; }, [&](auto const &...) { ++nelements; }, short_rows);
  EXPECT_EQ(nblocks, 0);
  EXPECT_EQ(nelements, short_rows.size());
}

TEST(SIMD, ScalarFallbackResults) {
  constexpr long n = 2 * simd::native_simd<double>::size + 3;
  array<double, 1> d(n);
  array<std::int64_t, 1> l(n);
  // An expression of another value type than the destination is evaluated element by element.
  for (long i = 0; i < n; ++i) { l(i) = i - 5; }
  d = l + 0;
  for (long i = 0; i < n; ++i) { EXPECT_EQ(d(i), double(i - 5)); }

  // frobenius_norm of arrays without SIMD loads (scalar nda::sum of nda::abs2)
  array<long double, 2> ldm(2, n);
  array<double, 2> c(n, n);
  ldm = 0.5L;
  c   = 0.5;
  EXPECT_EQ(frobenius_norm(ldm), std::sqrt(0.25L * 2 * n)); // long double result, exact sum of squares
  EXPECT_DOUBLE_EQ(frobenius_norm(c(range::all, range(0, n, 2))), std::sqrt(0.25 * n * ((n + 1) / 2)));
}
