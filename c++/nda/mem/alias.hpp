// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides memory bounds and conservative overlap checks for arrays and expressions.
 */

#pragma once

#include "../concepts.hpp"
#include "../macros.hpp"
#include "../traits.hpp"

#include <cstddef>
#include <cstdint>

#include <tuple>

namespace nda {

  template <char OP, Array A>
  struct expr_unary;
  template <char OP, ArrayOrScalar L, ArrayOrScalar R>
  struct expr;
  template <typename F, Array... A>
  struct expr_call;

} // namespace nda

namespace nda::mem {

  struct memory_bounds {
    std::uintptr_t begin;
    std::uintptr_t end;
  };

  /**
   * @brief Get the byte-address interval containing all elements of an array/view.
   *
   * @pre The view is nonempty, its displacements are representable, and its layout guarantees hold.
   * @details Uses compile-time contiguous/constant-stride guarantees where available. No elements are
   * dereferenced and no overflow checks are performed.
   */
  template <MemoryArray A>
  memory_bounds get_memory_bounds(A const &a) noexcept {
    constexpr std::uintptr_t itemsize = sizeof(get_value_t<A>);
    auto p = reinterpret_cast<std::uintptr_t>(a.data()); //NOLINT
    if constexpr (has_contiguous_layout<A>) {
      return {.begin = p, .end = p + static_cast<std::uintptr_t>(a.size()) * itemsize};
    } else {
      std::ptrdiff_t lo = 0, hi = 0;
      if constexpr (has_layout_strided_1d<A>) {
        // A constant linear stride needs only its final displacement.
        auto delta = (a.size() - 1) * a.indexmap().min_stride();
        (delta < 0 ? lo : hi) = delta;
      } else {
        // Each axis extends either the lower or the upper bound.
        for (int d = 0; d < get_rank<A>; ++d) {
          auto delta = (a.shape()[d] - 1) * a.indexmap().strides()[d];
          (delta < 0 ? lo : hi) += delta;
        }
      }
      // Unsigned arithmetic applies negative offsets modulo the address width.
      return {.begin = p + static_cast<std::uintptr_t>(lo) * itemsize,
              .end = p + static_cast<std::uintptr_t>(hi) * itemsize + itemsize};
    }
  }

  namespace detail {

    // Returns true for possible overlap other than exact in-place mappings.
    // The exact-mapping exemption assumes distinct destination elements.
    template <MemoryArray D>
    struct alias_checker {
      D const &dst;
      mem::memory_bounds bounds;

      template <typename T>
      bool operator()(T const &) const { return true; }

      template <Scalar S>
      bool operator()(S const &) const { return false; }

      template <MemoryArray A>
      bool operator()(A const &e) const {
        // Exact in-place operands introduce no dependencies between distinct destination elements.
        if constexpr (std::is_same_v<get_value_t<A>, get_value_t<D>> and get_rank<A> == get_rank<D>) {
          if (static_cast<void const *>(e.data()) == static_cast<void const *>(dst.data())
              and e.shape() == dst.shape() and e.indexmap().strides() == dst.indexmap().strides())
            return false;
        }
        auto other = mem::get_memory_bounds(e);
        return bounds.begin < other.end and other.begin < bounds.end;
      }

      template <char OP, Array A>
      bool operator()(expr_unary<OP, A> const &e) const { return (*this)(e.a); }

      template <char OP, ArrayOrScalar L, ArrayOrScalar R>
      bool operator()(expr<OP, L, R> const &e) const { return (*this)(e.l) or (*this)(e.r); }

      template <typename F, Array... A>
      bool operator()(expr_call<F, A...> const &e) const {
        return std::apply([&](auto const &...a) { return ((*this)(a) or ...); }, e.a);
      }
    };

  } // namespace detail

  /**
   * @brief Check for possible overlap, excluding exact in-place mappings.
   * @details Operands with the same element type, data pointer, shape and strides are exempt.
   * False means every explicit source operand is disjoint or exempt; it does not justify restrict.
   * @pre Destination elements are distinct when using the exemption for in-place SIMD.
   * @tparam A Destination memory array/view type.
   * @tparam B Source array or expression types.
   */
  template <MemoryArray A, Array... B>
  FORCEINLINE bool may_alias(A const &dst, B const &...src) {
    if constexpr (sizeof...(B) == 0) return false;
    else {
      if (dst.size() == 0) return false;
      // Calculate destination bounds once for all source operands.
      auto const checker = detail::alias_checker<A>{dst, get_memory_bounds(dst)};
      return ((src.size() != 0 and checker(src)) or ...);
    }
  }

} // namespace nda::mem
