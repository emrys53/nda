// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once

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
