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
      if (stride_order == 0) return i;                 // default C-order
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

    template <int I, uint64_t StaticExtents, uint64_t StrideOrder, size_t SIMD_SIZE, typename F_SIMD, typename F_SCALAR, size_t R,
              std::integral Int = long>
    FORCEINLINE void for_each_static_impl(std::array<Int, R> const &shape, std::array<long, R> &idxs, F_SIMD &f_simd, F_SCALAR &f_scalar) {
      // get the dimension over which to iterate and its extent
      static constexpr int J = index_from_stride_order<R>(StrideOrder, I);
      const long imax        = get_extent<J, R, StaticExtents>(shape);
      // Only difference from scalar implementation is that in the last dimension we call f_simd whenever we can.
      if constexpr (I == R - 1) {
        size_t i          = 0;
        const size_t ilim = imax & -static_cast<int64_t>(SIMD_SIZE);
        for (; i < ilim; i += SIMD_SIZE) {
          std::apply(f_simd, idxs);
          idxs[J] += SIMD_SIZE;
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
   * @tparam StrideOrder Encoded stride order.
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

  template <uint64_t StaticExtents, uint64_t StrideOrder, size_t SIMD_SIZE, typename F_SIMD, typename F_SCALAR, auto R, std::integral Int = long>
  FORCEINLINE void for_each_static(std::array<Int, R> const &shape, F_SIMD &&f_simd, F_SCALAR &&f_scalar) { // NOLINT (we do not want to forward here)
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

  template <size_t SIMD_SIZE, typename F_SIMD, typename F_SCALAR, auto R, std::integral Int = long>
  FORCEINLINE void for_each(std::array<Int, R> const &shape, F_SIMD &&f_simd, F_SCALAR &&f_scalar) { // NOLINT
    auto idxs = nda::stdutil::make_initialized_array<R>(0l);
    detail::for_each_static_impl<0, 0, 0, SIMD_SIZE, F_SIMD, F_SCALAR>(shape, idxs, f_simd, f_scalar);
  }

#ifdef NDA_HAVE_XSIMD
  /**
   * @brief Loop over all elements of one or more nda::Array objects of the same shape in SIMD-width blocks.
   *
   * @details All arrays must be contiguous with the same stride order and shape. If every array satisfies
   * nda::supports_flat_loop_v, the callables receive one nda::_linear_index_t per block (`f_simd`) or element
   * (`f_scalar`), otherwise a multi-index with `f_simd` stepping along the fastest dimension.
   *
   * @tparam F_SIMD Callable type applied to full SIMD blocks.
   * @tparam F_SCALAR Callable type applied to the remaining elements.
   * @tparam First nda::Array type of the first array.
   * @tparam Rest nda::Array types of the other arrays.
   * @param f_simd Callable applied to full SIMD blocks.
   * @param f_scalar Callable applied to the remaining elements.
   * @param first First array (determines the SIMD width, stride order and shape).
   * @param rest Other arrays.
   */
  template <typename F_SIMD, typename F_SCALAR, Array First, Array... Rest>
  FORCEINLINE void for_each_simd(F_SIMD &&f_simd, F_SCALAR &&f_scalar, First const &first, [[maybe_unused]] Rest const &...rest) { // NOLINT
    static_assert(Vectorizable<get_value_t<First>> and (Vectorizable<get_value_t<Rest>> and ...),
                  "Error in for_each_simd: All Array elements have to be vectorizable.");
    static_assert(has_contiguous_layout<First> and (has_contiguous_layout<Rest> and ...),
                  "Error in for_each_simd: all arrays must have a contiguous layout to be vectorized");
    static_assert(((get_layout_info<First>.stride_order == get_layout_info<Rest>.stride_order) and ...),
                  "Error in for_each_simd: all arrays must have the same stride order to be vectorized");
    static constexpr size_t simd_size = native_simd<get_value_t<First>>::size;
    static constexpr bool flat_loop   = supports_flat_loop_v<First> and (supports_flat_loop_v<Rest> and ...);
    EXPECTS(((first.shape() == rest.shape()) and ...));
    if constexpr (flat_loop) {
      const long n    = first.size();
      const long nlim = n & -simd_size;
      long k          = 0;
      for (; k < nlim; k += simd_size) { f_simd(_linear_index_t{k}); }
      for (; k < n; ++k) { f_scalar(_linear_index_t{k}); }
    } else {
      for_each_static<0, get_layout_info<First>.stride_order, simd_size>(first.shape(), f_simd, f_scalar);
    }
  }
#endif // NDA_HAVE_XSIMD

  /** @} */

} // namespace nda
