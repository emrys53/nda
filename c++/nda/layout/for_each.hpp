// Copyright (c) 2018--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides `for_each` functions for multi-dimensional arrays/views.
 */

#pragma once

#include "./permutation.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../simd/simd.hpp"
#include "../simd/simd_dispatch.hpp"
#include "../stdutil/array.hpp"
#include "../traits.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <utility>

namespace nda {

  /**
   * @addtogroup layout_utils
   * @{
   */

  namespace detail {

    // Get the i-th slowest moving dimension from a given encoded stride order.
    template <int R>
    constexpr int index_from_stride_order(uint64_t stride_order, int i) {
      // Unspecified (0) and undefined (-1) stride orders use C-order.
      if (stride_order == 0 or stride_order == uint64_t(-1)) return i;
      auto stride_order_arr = decode<R>(stride_order); // FIXME C++20
      return stride_order_arr[i];
    }

    // Get the extent of an array along its i-th dimension.
    template <int I, int R, uint64_t StaticExtents, std::integral Int = long>
    constexpr long get_extent(std::array<Int, R> const &shape) {
      if constexpr (StaticExtents == 0) {
        // dynamic extents
        return shape[I];
      } else {
        // full/partial static extents
        constexpr auto static_extents = decode<R>(StaticExtents); // FIXME C++20
        if constexpr (static_extents[I] == 0)
          return shape[I];
        else
          return static_extents[I];
      }
    }

    // Apply a callable object recursively to all possible index values of a given shape.
    template <int I, uint64_t StaticExtents, uint64_t StrideOrder, typename F, size_t R, std::integral Int = long>
    FORCEINLINE void for_each_static_impl(std::array<Int, R> const &shape, std::array<long, R> &idxs, F &f) {
      if constexpr (I == R) {
        // end of recursion
        std::apply(f, idxs);
      } else {
        // get the dimension over which to iterate and its extent
        static constexpr int J = index_from_stride_order<R>(StrideOrder, I);
        const long imax        = get_extent<J, R, StaticExtents>(shape);

        // loop over all indices of the current dimension
        for (long i = 0; i < imax; ++i) {
          // recursive call for the next dimension
          for_each_static_impl<I + 1, StaticExtents, StrideOrder>(shape, idxs, f);
          ++idxs[J];
        }
        idxs[J] = 0;
      }
    }

    template <int I, uint64_t StaticExtents, uint64_t StrideOrder, size_t SIMD_SIZE, size_t R, std::integral Int = long>
    FORCEINLINE void for_each_static_impl(std::array<Int, R> const &shape, std::array<long, R> &idxs, auto &f_simd, auto &f_scalar) {
      // get the dimension over which to iterate and its extent
      static constexpr int J = index_from_stride_order<R>(StrideOrder, I);
      const long imax        = get_extent<J, R, StaticExtents>(shape);
      // Only difference from scalar implementation is that in the last dimension we call f_simd whenever we can.
      if constexpr (I == R - 1) {
        constexpr long w = SIMD_SIZE;
        const long ilim  = imax & -w;
        long i           = 0;
        for (; i < ilim; i += w) {
          std::apply(f_simd, idxs);
          idxs[J] += w;
        }
        for (; i < imax; ++i) {
          std::apply(f_scalar, idxs);
          ++idxs[J];
        }
        idxs[J] = 0;
      } else {
        // loop over all indices of the current dimension
        for (long i = 0; i < imax; ++i) {
          // recursive call for the next dimension
          for_each_static_impl<I + 1, StaticExtents, StrideOrder, SIMD_SIZE>(shape, idxs, f_simd, f_scalar);
          ++idxs[J];
        }
        idxs[J] = 0;
      }
    }

  } // namespace detail

  /**
   * @brief Loop over all possible index values of a given shape and apply a function to them.
   *
   * @details It traverses all possible indices in the order given by the encoded `StrideOrder` parameter. The shape is
   * either specified at runtime (`StaticExtents == 0`) or, partially or fully, at compile time (`StaticExtents != 0`).
   * The given function `f` must be callable with as many `long` values as the number of dimensions in the shape array,
   * e.g. for a 3-dimensional shape the function must be callable as `f(long, long, long)`.
   *
   * @tparam StaticExtents Encoded static extents.
   * @tparam StrideOrder Encoded stride order. Zero or uint64_t(-1) selects C-order.
   * @tparam F Callable type.
   * @tparam R Number of dimensions.
   * @tparam Int Integer type used in the shape array.
   *
   * @param shape Shape to loop over (index bounds).
   * @param f Callable object.
   */
  template <uint64_t StaticExtents, uint64_t StrideOrder, typename F, auto R, std::integral Int = long>
  FORCEINLINE void for_each_static(std::array<Int, R> const &shape, F &&f) { // NOLINT (we do not want to forward here)
    auto idxs = nda::stdutil::make_initialized_array<R>(0l);
    detail::for_each_static_impl<0, StaticExtents, StrideOrder>(shape, idxs, f);
  }

  /**
   * @brief Loop over all possible index values of a given shape in blocks along the fastest dimension.
   *
   * @details Same traversal as for_each_static(std::array<Int, R> const &, F &&), except that the innermost loop (over
   * the fastest dimension of the stride order) advances by `SIMD_SIZE` indices at a time: `f_simd(idx...)` is called
   * with the multi-index of the first element of every full block, and `f_scalar(idx...)` for every remaining index
   * of that dimension. Both callables take as many `long` values as there are dimensions.
   *
   * @tparam StaticExtents Encoded static extents.
   * @tparam StrideOrder Encoded stride order. Zero or uint64_t(-1) selects C-order.
   * @tparam SIMD_SIZE Number of indices in a block.
   * @tparam R Number of dimensions.
   * @tparam Int Integer type used in the shape array.
   *
   * @param shape Shape to loop over (index bounds).
   * @param f_simd Callable applied to the first index of every full block.
   * @param f_scalar Callable applied to every index not covered by a block.
   */
  template <uint64_t StaticExtents, uint64_t StrideOrder, size_t SIMD_SIZE, auto R, std::integral Int = long>
  FORCEINLINE void for_each_static(std::array<Int, R> const &shape, auto &&f_simd, auto &&f_scalar) { // NOLINT (we do not want to forward here)
    auto idxs = nda::stdutil::make_initialized_array<R>(0l);
    detail::for_each_static_impl<0, StaticExtents, StrideOrder, SIMD_SIZE>(shape, idxs, f_simd, f_scalar);
  }

  /**
   * @brief Loop over all possible index values of a given shape and apply a function to them.
   *
   * @details It traverses all possible indices in C-order, i.e. the last index varies the fastest. The shape is
   * specified at runtime and the given function `f` must be callable with as many `long` values as the number of
   * dimensions in the shape array, e.g. for a 3-dimensional shape the function must be callable as
   * `f(long, long, long)`.
   *
   * @tparam F Callable type.
   * @tparam R Number of dimensions.
   * @tparam Int Integer type used in the shape array.
   *
   * @param shape Shape to loop over (index bounds).
   * @param f Callable object.
   */
  template <typename F, auto R, std::integral Int = long>
  FORCEINLINE void for_each(std::array<Int, R> const &shape, F &&f) { // NOLINT (we do not want to forward here)
    auto idxs = nda::stdutil::make_initialized_array<R>(0l);
    detail::for_each_static_impl<0, 0, 0>(shape, idxs, f);
  }

  /**
   * @brief Constexpr variable that is true if nda::for_each_simd loops over arrays of the given types in SIMD blocks.
   *
   * @details True if `First` is SIMD-enabled (nda::is_simd_enabled_v), every other array is SIMD-enabled for the value
   * type of `First` and all stride orders agree. Always false without xsimd.
   *
   * @tparam First nda::Array type of the first array.
   * @tparam Rest nda::Array types of the other arrays.
   */
  template <typename First, typename... Rest>
  inline constexpr bool has_simd_loop_v = is_simd_enabled_v<First> and (is_simd_enabled_v<Rest, get_value_t<First>> and ...)
     and ((get_layout_info<First>.stride_order == get_layout_info<Rest>.stride_order) and ...);

  /**
   * @brief Loop over all elements of one or more nda::Array objects of the same shape in SIMD-width blocks.
   *
   * @details The callables are called as `f(first, rest..., idx...)`. The loop shape is decided at compile time:
   * - if the arrays can be looped over in SIMD blocks (nda::has_simd_loop_v), `f_simd` is called for every full block
   *   of `nda::simd::native_simd<value_type>::size` elements and `f_scalar` for the remaining ones; otherwise (always
   *   without xsimd) `f_scalar` is called for every element, in the stride order of `first`, and `f_simd` is never
   *   instantiated;
   * - if all arrays are contiguous and elementwise (nda::is_elementwise_v), `idx...` is a single
   *   nda::_linear_index_t and the tail is at most one block; otherwise `idx...` is the multi-index and the blocks
   *   follow the fastest dimension without crossing a row, so every row has its own tail and a row shorter than a
   *   block is visited by `f_scalar` only.
   *
   * @code{.cpp}
   * nda::array<double, 2> a(16, 16), b(16, 16);
   * // b = 2.0 * a + 1.0, evaluated in blocks where possible
   * nda::for_each_simd([](auto &dst, auto const &src, auto const &...i) { dst.store(src.load(i...), i...); },
   *                    [](auto &dst, auto const &src, auto const &...i) { dst(i...) = src(i...); }, b, 2.0 * a + 1.0);
   * @endcode
   *
   * @tparam First nda::Array type of the first array.
   * @tparam Rest nda::Array types of the other arrays.
   * @param f_simd Callable applied to full SIMD blocks.
   * @param f_scalar Callable applied to the remaining elements.
   * @param first First array (determines the value type, SIMD width, stride order and shape).
   * @param rest Other arrays.
   */
  template <Array First, Array... Rest>
  // NOLINTNEXTLINE (we do not want to forward here)
  FORCEINLINE void for_each_simd([[maybe_unused]] auto &&f_simd, auto &&f_scalar, First &&first, Rest &&...rest) {
    EXPECTS(((first.shape() == rest.shape()) and ...));
    constexpr auto lhs_stride_order = get_layout_info<First>.stride_order;
    if constexpr (not has_simd_loop_v<First, Rest...>) {
      for_each_static<0, lhs_stride_order>(first.shape(), [&](auto const &...idx) { f_scalar(first, rest..., idx...); });
    } else {
#ifdef NDA_HAVE_XSIMD
      constexpr long simd_size = simd::native_simd<get_value_t<First>>::size;
#else
      constexpr long simd_size = 1; // never used: without xsimd no array is SIMD-enabled
#endif
      constexpr auto flat_loop = has_contiguous_layout<First> and (has_contiguous_layout<Rest> and ...)
         and is_elementwise_v<First> and (is_elementwise_v<Rest> and ...);
      if constexpr (flat_loop) {
        const long n    = first.size();
        const long nlim = n & -simd_size;
        long k          = 0;
        for (; k < nlim; k += simd_size) { f_simd(first, rest..., _linear_index_t{k}); }
        for (; k < n; ++k) { f_scalar(first, rest..., _linear_index_t{k}); }
      } else {
        for_each_static<0, lhs_stride_order, simd_size>(
           first.shape(), [&](auto const &...idx) { f_simd(first, rest..., idx...); }, [&](auto const &...idx) { f_scalar(first, rest..., idx...); });
      }
    }
  }

  /** @} */

} // namespace nda
