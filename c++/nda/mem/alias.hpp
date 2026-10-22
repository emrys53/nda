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

  /// Byte-address interval `[begin, end)` of a block of memory.
  struct memory_bounds {
    std::uintptr_t begin;
    std::uintptr_t end;
  };

  /**
   * @brief Get the byte-address interval containing all elements of an array/view.
   *
   * @details Contiguous layouts need only the size, constant-stride layouts (nda::has_layout_strided_1d) only the
   * displacement of the last element, and other layouts sum the negative and the positive displacement of each axis
   * separately. No element is dereferenced and no overflow check is performed. An empty array/view gives the empty
   * interval `[p, p)` at its data pointer `p`.
   *
   * @pre The displacements of the elements are representable and the compile-time layout guarantees hold.
   * @tparam A nda::MemoryArray type.
   * @param a Array/view.
   * @return nda::mem::memory_bounds of the elements.
   */
  template <MemoryArray A>
  [[nodiscard]] memory_bounds get_memory_bounds(A const &a) noexcept {
    constexpr std::uintptr_t itemsize = sizeof(get_value_t<A>);
    auto p = reinterpret_cast<std::uintptr_t>(a.data()); //NOLINT
    if constexpr (has_contiguous_layout<A>) {
      return {.begin = p, .end = p + static_cast<std::uintptr_t>(a.size()) * itemsize};
    } else {
      if (a.size() == 0) return {.begin = p, .end = p};
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

  /// How the sources of an assignment share memory with its destination.
  enum class aliasing {
    /// No source shares memory with the destination.
    disjoint,
    /// Memory is shared only with exact in-place operands (same element type, data pointer, shape and strides as the
    /// destination), e.g. `a` in `a = a + 1`: every element is read where it is written (if no stride is zero).
    in_place,
    /// A source may share memory with the destination in any other way.
    may_overlap
  };

  /// Aliasing of two sources together: `may_overlap` if either may overlap, otherwise `in_place` if either is an
  /// in-place operand, otherwise `disjoint`.
  FORCEINLINE constexpr aliasing operator|(aliasing a, aliasing b) {
    if (a == aliasing::may_overlap or b == aliasing::may_overlap) return aliasing::may_overlap;
    if (a == aliasing::in_place or b == aliasing::in_place) return aliasing::in_place;
    return aliasing::disjoint;
  }

  namespace detail {

    // Classifies the array leaves of a source relative to the destination (see nda::mem::detect_aliasing).
    template <MemoryArray D>
    struct alias_checker {
      D const &dst; //NOLINT
      mem::memory_bounds bounds;

      template <typename T>
      FORCEINLINE aliasing operator()(T const &) const { return aliasing::may_overlap; }

      template <Scalar S>
      FORCEINLINE aliasing operator()(S const &) const { return aliasing::disjoint; }

      template <MemoryArray A>
      FORCEINLINE aliasing operator()(A const &e) const {
        if (e.size() == 0) return aliasing::disjoint;
        if constexpr (std::is_same_v<get_value_t<A>, get_value_t<D>> and get_rank<A> == get_rank<D>) {
          if (static_cast<void const *>(e.data()) == static_cast<void const *>(dst.data())
              and e.shape() == dst.shape() and e.indexmap().strides() == dst.indexmap().strides())
            return aliasing::in_place;
        }
        auto other = mem::get_memory_bounds(e);
        return bounds.begin < other.end and other.begin < bounds.end ? aliasing::may_overlap : aliasing::disjoint;
      }

      template <char OP, Array A>
      FORCEINLINE aliasing operator()(expr_unary<OP, A> const &e) const { return (*this)(e.a); }

      template <char OP, ArrayOrScalar L, ArrayOrScalar R>
      FORCEINLINE aliasing operator()(expr<OP, L, R> const &e) const { return (*this)(e.l) | (*this)(e.r); }

      template <typename F, Array... A>
      FORCEINLINE aliasing operator()(expr_call<F, A...> const &e) const {
        return std::apply([&](auto const &...a) { return (aliasing::disjoint | ... | (*this)(a)); }, e.a);
      }
    };

  } // namespace detail

  /**
   * @brief Detect how the sources of an assignment share memory with its destination.
   *
   * @details Compares the byte-address interval of the destination (nda::mem::get_memory_bounds) with the intervals of
   * the memory arrays in the sources. Conservative: `may_overlap` also covers interleaved views that share no element
   * and sources that are neither memory arrays, expressions nor scalars.
   *
   * @tparam A nda::MemoryArray type of the destination.
   * @tparam B nda::Array types of the sources.
   * @param dst Destination array/view.
   * @param src Source arrays, views or expressions.
   * @return nda::mem::aliasing of the sources.
   */
  template <MemoryArray A, Array... B>
  [[nodiscard]] FORCEINLINE aliasing detect_aliasing(A const &dst, B const &...src) {
    if (dst.size() == 0) return aliasing::disjoint;
    // Calculate destination bounds once for all source operands.
    [[maybe_unused]] auto const checker = detail::alias_checker<A>{dst, get_memory_bounds(dst)};
    return (aliasing::disjoint | ... | checker(src));
  }

} // namespace nda::mem
