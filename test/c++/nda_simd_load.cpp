// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./simd_test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/simd/mock_simd.hpp>

using namespace nda;

namespace {

  template <typename E>
  void check_load(E const &expression, long offset = 0) {
    std::array<long, get_rank<E>> index{};
    constexpr auto axis = decode<get_rank<E>>(get_layout_info<E>.stride_order).back();
    index[axis] = offset;
    auto batch = std::apply([&](auto... i) { return expression.load(i...); }, index);
    for (size_t lane = 0; lane < decltype(batch)::size; ++lane) {
      std::apply([&](auto... i) { expect_simd_eq(batch.get(lane), expression(i...)); }, index);
      ++index[axis];
    }
  }

  template <typename T, int Rank, typename Layout, char Algebra>
  void check_load_store() {
    constexpr long w = native_simd<T>::size;
    auto const shape = stdutil::make_initialized_array<Rank>(32l);
    using array_t = basic_array<T, Rank, Layout, Algebra, heap<>>;
    array_t a(shape);
    fill_random(a, 1, 0, 1000);
    check_load(a);
    check_load(a, w);

    auto load_at  = [&](std::array<long, Rank> const &idx) { return std::apply([&](auto... i) { return a.load(i...); }, idx); };
    auto store_at = [&](auto const &batch, std::array<long, Rank> const &idx) { std::apply([&](auto... i) { a.store(batch, i...); }, idx); };

    // store the sum of the first two blocks of the fastest dimension into the first one, the second stays untouched
    int const fastest = a.stride_order().back();
    std::array<long, Rank> first{}, second{};
    second[fastest] = w;
    auto const x = load_at(first), y = load_at(second);
    auto const sum = x + y;
    store_at(sum, first);
    for (long lane = 0; lane < w; ++lane, ++first[fastest], ++second[fastest]) {
      std::apply([&](auto... i) { expect_simd_eq(a(i...), sum.get(lane)); }, first);
      std::apply([&](auto... i) { expect_simd_eq(a(i...), y.get(lane)); }, second);
    }

    // a block running past the end of the fastest dimension throws (the tests enforce bound checks)
    std::array<long, Rank> last{};
    last[fastest] = 32 - w;
    EXPECT_NO_THROW(load_at(last));
    ++last[fastest];
    EXPECT_THROW(load_at(last), nda::runtime_error);
    EXPECT_THROW(store_at(x, last), nda::runtime_error);
    EXPECT_NO_THROW(a.load(_linear_index_t{a.size() - w}));
    EXPECT_THROW(a.load(_linear_index_t{a.size() - w + 1}), nda::runtime_error);
    EXPECT_THROW(a.store(x, _linear_index_t{a.size() - w + 1}), nda::runtime_error);

    array_t b(shape), c(shape);
    fill_random(b, 2, 0, 1000);
    fill_random(c, 3, 1, 1000); // a divisor
    check_load(b + c);
    check_load(b - c);
    if constexpr (Algebra == 'A') {
      check_load(b * c);
      check_load(b / c);
    }
  }

  template <typename T>
  struct scalar_f {
    template <typename... Args>
    T operator()(Args const &...args) const { return (args + ...) + T{1}; }
  };

  template <typename T>
  struct mock_f : simd::mock_simd<mock_f<T>, T>, scalar_f<T> {
    using scalar_f<T>::operator();
  };

  template <typename T, int Rank, typename Layout>
  void check_mock() {
    auto const shape = stdutil::make_initialized_array<Rank>(32l);
    array<T, Rank, Layout> a(shape), b(shape), c(shape);
    fill_random(a, 1, 0, 1000);
    fill_random(b, 2, 0, 1000);
    fill_random(c, 3, 0, 1000);
    auto check = [&](auto const &...inputs) {
      auto mocked = map(mock_f<T>{})(inputs...);
      auto scalar = map(scalar_f<T>{})(inputs...);
      check_load(mocked);
      EXPECT_ARRAY_NEAR(mocked, scalar);
    };
    check(a);
    check(a, b);
    check(a, b, c);
  }

} // namespace

template <typename T>
class SIMDLoad : public ::testing::Test {};

TYPED_TEST_SUITE(SIMDLoad, simd_types);

// Rank 1 has a single stride order, so only the higher ranks run with both layouts.
TYPED_TEST(SIMDLoad, LoadStore) {
  check_load_store<TypeParam, 1, C_layout, 'A'>();
  check_load_store<TypeParam, 2, C_layout, 'A'>();
  check_load_store<TypeParam, 2, F_layout, 'A'>();
  check_load_store<TypeParam, 3, C_layout, 'A'>();
  check_load_store<TypeParam, 3, F_layout, 'A'>();
  check_load_store<TypeParam, 2, C_layout, 'M'>();
  check_load_store<TypeParam, 2, F_layout, 'M'>();
}

TYPED_TEST(SIMDLoad, Mock) {
  check_mock<TypeParam, 1, C_layout>();
  check_mock<TypeParam, 3, C_layout>();
  check_mock<TypeParam, 3, F_layout>();
}
