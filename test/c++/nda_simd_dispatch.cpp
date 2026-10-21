// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./simd_test_common.hpp"

using namespace nda;

namespace {

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

// The expression tests use abs, which has no overload for unsigned types.
template <typename T>
class SIMDExpressions : public ::testing::Test {};
using signed_simd_types = ::testing::Types<std::int32_t, std::int64_t, float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SIMDExpressions, signed_simd_types);

TYPED_TEST(SIMDExpressions, FlatLoop) {
  check_flat_loop<TypeParam, C_layout>();
  check_flat_loop<TypeParam, F_layout>();
}
