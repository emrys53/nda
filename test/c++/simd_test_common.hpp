// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once

#include "./test_common.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

using simd_types = ::testing::Types<std::int32_t, std::int64_t, std::uint32_t, std::uint64_t, float, double,
                                  std::complex<float>, std::complex<double>>;

// Lane count of the native batch of T, or 8 without one, so that size-dependent tests also run on the scalar path.
template <typename T>
constexpr long simd_width() {
#ifdef NDA_HAVE_XSIMD
  if constexpr (nda::Vectorizable<T>) { return static_cast<long>(nda::native_simd<T>::size); }
#endif
  return 8;
}

template <typename A, typename B>
void fill_random(A &a, unsigned seed, B lo, B hi) {
  using T      = nda::get_value_t<A>;
  using real_t = nda::remove_complex_t<T>;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(lo, hi);
  auto draw = [&] {
    double const x = dist(gen);
    return static_cast<real_t>(std::is_integral_v<B> ? std::floor(x) : x);
  };
  for (auto &x : a) {
    if constexpr (nda::is_complex_v<T>) {
      auto const re = draw();
      x             = T(re, draw());
    } else {
      x = draw();
    }
  }
}

template <typename T>
void expect_simd_eq(T actual, T expected) {
  if constexpr (nda::is_complex_v<T>) {
    // complex results agree up to rounding in norm, not per component: a small imaginary part of a quotient can differ
    // by many ULPs between two correctly rounded divisions
    using real_t = nda::remove_complex_t<T>;
    EXPECT_LE(std::abs(actual - expected), 4 * std::numeric_limits<real_t>::epsilon() * std::abs(expected));
  } else if constexpr (std::same_as<T, float>) {
    EXPECT_FLOAT_EQ(actual, expected);
  } else if constexpr (std::same_as<T, double>) {
    EXPECT_DOUBLE_EQ(actual, expected);
  } else {
    EXPECT_EQ(actual, expected);
  }
}
