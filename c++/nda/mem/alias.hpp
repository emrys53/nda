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
    /// Address of the first byte.
    std::uintptr_t begin;
    /// Address one past the last byte.
    std::uintptr_t end;
  };

  /**
   * @brief Get the byte-address interval containing all elements of an array/view.
   *
   * @details Contiguous layouts need only the size, constant-stride layouts (nda::has_layout_strided_1d) only the
   * displacement of the last element, and other layouts sum the negative and the positive displacement of each axis
   * separately. No element is dereferenced and no overflow check is performed.
   *
   * @pre The view is nonempty, its displacements are representable, and its compile-time layout guarantees hold.
   * @tparam A nda::MemoryArray type.
   * @param a Array/view.
   * @return nda::mem::memory_bounds of the elements.
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
   * @brief Check whether writing the elements of a destination array/view may change what the given source arrays
   * or expressions read, except for exact in-place mappings.
   *
   * @details The byte-address interval of the destination (nda::mem::get_memory_bounds) is compared with the one of
   * every memory array found in the sources: the array leaves of nda::expr_unary, nda::expr and nda::expr_call
   * (scalar operands never alias; other nda::Array types are conservatively reported as aliasing). A source with the
   * same element type, data pointer, shape and strides as the destination is exempt: it reads every element exactly
   * where it is written, so an evaluation that reads a block of elements before writing it is still correct, e.g.
   * `a = a + 1`. The exemption assumes that the destination elements are distinct (no two indices map to the same
   * address); a view with a zero stride breaks it.
   *
   * The result is conservative: `true` includes interleaved views that share no element (the even and the odd
   * elements of one buffer); `false` means that every source is either disjoint or an exact in-place operand, not
   * that the operands do not overlap, so it must not be used to justify `restrict`. Empty destinations and empty
   * sources never alias.
   *
   * @tparam A nda::MemoryArray type of the destination.
   * @tparam B nda::Array types of the sources (arrays, views or expressions).
   * @param dst Destination array/view.
   * @param src Source arrays, views or expressions.
   * @return True if a source may overlap the destination other than as an exact in-place mapping.
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
