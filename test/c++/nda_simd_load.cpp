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
    constexpr long w = simd::native_simd<T>::size;
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

    // the block at a linear index holds the elements at that offset of the flat data; store the double of the second
    // block into the first one, the second stays untouched
    auto const z = a.load(_linear_index_t{w});
    for (long lane = 0; lane < w; ++lane) expect_simd_eq(z.get(lane), a(_linear_index_t{w + lane}));
    a.store(z + z, _linear_index_t{0});
    for (long lane = 0; lane < w; ++lane) {
      expect_simd_eq(a(_linear_index_t{lane}), z.get(lane) + z.get(lane));
      expect_simd_eq(a(_linear_index_t{w + lane}), z.get(lane));
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
    fill_random(c, 3, 1, 1000);
    check_load(b + c);
    check_load(b - c);
    if constexpr (Algebra == 'A') {
      check_load(b * c);
      check_load(b / c);
    }
  }

  template <typename T>
  struct scalar_f {
    T operator()(auto const &...args) const { return (args + ...) + T{1}; }
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
      EXPECT_ARRAY_EQ(make_regular(mocked), make_regular(scalar)); // lane-by-lane scalar calls give identical results
    };
    check(a);
    check(a, b);
    check(a, b, c);
  }

  // A non-contiguous view with a unit stride in its fastest dimension: the middle plane of a rank-3 array. Blocks
  // follow the fastest dimension of the view and must not touch the neighbouring planes.
  template <typename T, typename Layout>
  void check_view_load_store() {
    constexpr long w = simd::native_simd<T>::size;
    array<T, 3, Layout> t(32, 3, 32);
    fill_random(t, 1, 0, 1000);
    auto const reference = t;
    auto v               = t(range::all, 1, range::all);
    static_assert(has_layout_smallest_stride_is_one<decltype(v)> and not has_contiguous_layout<decltype(v)>);
    check_load(v);
    check_load(v, w);

    auto load_at  = [&](std::array<long, 2> const &idx) { return std::apply([&](auto... i) { return v.load(i...); }, idx); };
    auto store_at = [&](auto const &batch, std::array<long, 2> const &idx) { std::apply([&](auto... i) { v.store(batch, i...); }, idx); };

    // store the sum of the first two blocks of the fastest dimension into the first one
    int const fastest = decode<2>(get_layout_info<decltype(v)>.stride_order).back();
    std::array<long, 2> first{0, 0};
    std::array<long, 2> const second = (fastest == 0 ? std::array<long, 2>{w, 0} : std::array<long, 2>{0, w});
    auto const sum = load_at(first) + load_at(second);
    store_at(sum, first);
    for (long lane = 0; lane < w; ++lane, ++first[fastest]) { std::apply([&](auto... i) { expect_simd_eq(v(i...), sum.get(lane)); }, first); }

    // the planes next to the view are untouched
    for (long i = 0; i < 32; ++i)
      for (long k = 0; k < 32; ++k) {
        expect_simd_eq(t(i, 0, k), reference(i, 0, k));
        expect_simd_eq(t(i, 2, k), reference(i, 2, k));
      }

    // a block running past the end of the fastest dimension of the view throws
    std::array<long, 2> last{};
    last[fastest] = 32 - w + 1;
    EXPECT_THROW(load_at(last), nda::runtime_error);
  }

} // namespace

template <typename T>
class SIMDLoad : public ::testing::Test {};

TYPED_TEST_SUITE(SIMDLoad, simd_types);

// Rank 1 has a single stride order, so only rank 2 runs with both layouts; the views are planes of rank-3 arrays.
TYPED_TEST(SIMDLoad, LoadStore) {
  check_load_store<TypeParam, 1, C_layout, 'A'>();
  check_load_store<TypeParam, 2, C_layout, 'A'>();
  check_load_store<TypeParam, 2, F_layout, 'A'>();
  check_load_store<TypeParam, 2, C_layout, 'M'>();
  check_load_store<TypeParam, 2, F_layout, 'M'>();
  check_view_load_store<TypeParam, C_layout>();
  check_view_load_store<TypeParam, F_layout>();
}

TYPED_TEST(SIMDLoad, Mock) {
  check_mock<TypeParam, 1, C_layout>();
  check_mock<TypeParam, 3, C_layout>();
  check_mock<TypeParam, 3, F_layout>();
}
