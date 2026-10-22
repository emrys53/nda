// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include <nda/mem/alias.hpp>
#include "./test_common.hpp"

#include <array>
#include <complex>
#include <cstdint>

using nda::mem::aliasing;
using nda::mem::detect_aliasing;

TEST(NDA, MemoryBounds) {
  auto check = [](auto const &a, auto *begin, auto *end) {
    auto bounds = nda::mem::get_memory_bounds(a);
    EXPECT_EQ(bounds.begin, reinterpret_cast<std::uintptr_t>(begin));
    EXPECT_EQ(bounds.end, reinterpret_cast<std::uintptr_t>(end));
  };
  nda::array<double, 2> a(3, 7);
  nda::array<std::complex<double>, 2, nda::F_layout> z(3, 7);
  check(a, a.data(), a.data() + a.size());
  check(z, z.data(), z.data() + z.size());

  std::array<double, 64> storage{};
  using linear_layout = nda::basic_layout<0, nda::C_stride_order<2>, nda::layout_prop_e::strided_1d>;
  using linear_view = nda::array_view<double, 2, linear_layout>;
  linear_view forward{linear_view::layout_t{{3, 7}, {14, 2}}, storage.data()};
  linear_view backward{linear_view::layout_t{{3, 7}, {-14, -2}}, storage.data() + 40};
  nda::array_view<double, 2> mixed{{{3, 7}, {10, -2}}, storage.data() + 16};
  check(forward, storage.data(), storage.data() + 41);
  check(backward, storage.data(), storage.data() + 41);
  check(mixed, storage.data() + 4, storage.data() + 37);

  // empty views give an empty interval at their data pointer
  check(linear_view{linear_view::layout_t{{0, 7}, {14, 2}}, storage.data() + 8}, storage.data() + 8, storage.data() + 8);
  check(nda::array_view<double, 2>{{{3, 0}, {10, -2}}, storage.data() + 16}, storage.data() + 16, storage.data() + 16);
}

TEST(NDA, MemoryAliasViews) {
  std::array<double, 32> storage{};
  using view_t = nda::array_view<double, 1>;
  view_t head{{8}, storage.data()}, shifted{{8}, storage.data() + 1}, adjacent{{8}, storage.data() + 8};
  view_t shorter{{7}, storage.data()}, reverse{view_t::layout_t{{8}, {-1}}, storage.data() + 7};
  view_t even{view_t::layout_t{{8}, {2}}, storage.data()}, odd{view_t::layout_t{{8}, {2}}, storage.data() + 1};
  EXPECT_EQ(detect_aliasing(head, head), aliasing::in_place); // Only the exact mapping is in place, not other shapes or strides.
  EXPECT_EQ(detect_aliasing(head, shorter), aliasing::may_overlap);
  EXPECT_EQ(detect_aliasing(head, even), aliasing::may_overlap);
  EXPECT_EQ(detect_aliasing(head, reverse), aliasing::may_overlap);
  EXPECT_EQ(detect_aliasing(head, adjacent), aliasing::disjoint);
  EXPECT_EQ(detect_aliasing(even, odd), aliasing::may_overlap); // Conservative for interleaved elements.
  EXPECT_EQ(detect_aliasing(head, head + shifted), aliasing::may_overlap); // Exact operands must not hide shifted overlaps.
}

TEST(NDA, MemoryAliasExpressions) {
  nda::array<double, 1> a(8), b(8), dst(8), empty(0);
  int calls = 0;
  auto mapped = nda::map([&](double x, double y) { ++calls; return x + y; })(a, b);
  EXPECT_EQ(detect_aliasing(dst, -a + 2.0 * mapped, b), aliasing::disjoint);
  EXPECT_EQ(detect_aliasing(a, mapped), aliasing::in_place); // a is an exact operand, b is disjoint
  EXPECT_EQ(detect_aliasing(empty, mapped), aliasing::disjoint);
  EXPECT_EQ(detect_aliasing(dst, empty + 1.0), aliasing::disjoint);
  EXPECT_EQ(detect_aliasing(dst), aliasing::disjoint);
}
