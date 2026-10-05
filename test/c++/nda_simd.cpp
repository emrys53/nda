// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <cstdint>

using simd_types = ::testing::Types<std::int32_t, std::int64_t, std::uint32_t, std::uint64_t, float, double,
                                  std::complex<float>, std::complex<double>>;

template <typename T>
void expect_simd_eq(T actual, T expected) {
  if constexpr (nda::is_complex_v<T>) {
    expect_simd_eq(actual.real(), expected.real());
    expect_simd_eq(actual.imag(), expected.imag());
  } else if constexpr (std::same_as<T, float>) {
    EXPECT_FLOAT_EQ(actual, expected);
  } else if constexpr (std::same_as<T, double>) {
    EXPECT_DOUBLE_EQ(actual, expected);
  } else {
    EXPECT_EQ(actual, expected);
  }
}


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
    std::array<long, Rank> shape;
    shape.fill(32);
    using array_t = basic_array<T, Rank, Layout, Algebra, heap<>>;
    array_t a = rand(shape);
    check_load(a);
    check_load(a, native_simd<T>::size);

    std::array<long, Rank> first{}, second{};
    second[a.stride_order().back()] = native_simd<T>::size;
    auto x = std::apply([&](auto... i) { return a.load(i...); }, first);
    auto y = std::apply([&](auto... i) { return a.load(i...); }, second);
    auto sum = x + y;
    std::apply([&](auto... i) { a.store(sum, i...); }, first);
    for (size_t lane = 0; lane < decltype(sum)::size; ++lane) {
      std::apply([&](auto... i) { expect_simd_eq(a(i...), sum.get(lane)); }, first);
      ++first[a.stride_order().back()];
    }

    array_t b = rand(shape), c = rand(shape);
    for (auto &value : c) {
      if (value == T{0}) { value = T{1}; }
    }
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
    std::array<long, Rank> shape;
    shape.fill(32);
    array<T, Rank, Layout> a = rand(shape), b = rand(shape), c = rand(shape);
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

  template <typename T, typename Layout>
  void check_layout() {
    check_load_store<T, 1, Layout, 'A'>();
    check_load_store<T, 2, Layout, 'A'>();
    check_load_store<T, 3, Layout, 'A'>();
    check_load_store<T, 2, Layout, 'M'>();
    check_load_store<T, 1, Layout, 'V'>();
  }

} // namespace

template <typename T>
class SIMDLoad : public ::testing::Test {};

TYPED_TEST_SUITE(SIMDLoad, simd_types);

TYPED_TEST(SIMDLoad, LoadStore) {
  check_layout<TypeParam, C_layout>();
  check_layout<TypeParam, F_layout>();
}

TYPED_TEST(SIMDLoad, Mock) {
  check_mock<TypeParam, 1, C_layout>();
  check_mock<TypeParam, 2, C_layout>();
  check_mock<TypeParam, 3, C_layout>();
  check_mock<TypeParam, 1, F_layout>();
  check_mock<TypeParam, 2, F_layout>();
  check_mock<TypeParam, 3, F_layout>();
}
