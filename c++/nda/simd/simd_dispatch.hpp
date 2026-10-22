// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides compile-time traits that decide whether an expression is evaluated with native SIMD loads.
 */

#pragma once

#include "../concepts.hpp"
#include "./simd_concepts.hpp"

#include <type_traits>

namespace nda {

  /**
   * @addtogroup simd
   * @{
   */

  /**
   * @brief Constexpr variable that is true if an expression of type `A` can be loaded as `nda::simd::native_simd<T>`
   * batches using only native SIMD loads.
   *
   * @details False by default. It is specialized next to the node types of an expression tree, and every node must
   * produce the value type `T` it is asked for:
   * - nda::basic_array, nda::basic_array_view: `T` must be nda::simd::Vectorizable,
   * - nda::expr_unary, nda::expr: the operands are checked with the same `T` (a scalar operand is broadcast as `T` and
   *   must be convertible to it),
   * - nda::expr_call: may change the value type (e.g. `abs` of a complex array); the arguments are checked with their
   *   own value types, all batches must have the same width (nda::simd::SimdPreservesLaneWidth) and `F::load` must map
   *   the argument batches to a batch of `T` (nda::simd::LoadWithNativeSimdTo).
   *
   * @tparam A Type of the expression.
   * @tparam T Scalar type the expression has to produce, without cv-qualifiers (defaults to its own value type).
   */
  template <typename A, typename T = get_value_t<A>>
  inline constexpr bool has_simd_load_v = false;

  // Specialization of nda::has_simd_load_v for cvref types.
  template <typename A, typename T>
    requires(not std::is_same_v<A, std::remove_cvref_t<A>>)
  inline constexpr bool has_simd_load_v<A, T> = has_simd_load_v<std::remove_cvref_t<A>, T>;

  /**
   * @brief Constexpr variable that is true if an expression of type `A` is evaluated into values of type `T` with
   * native SIMD loads.
   *
   * @details The loads read `native_simd<T>::size` consecutive elements along the fastest dimension, so the
   * expression must have a unit stride there (nda::has_layout_smallest_stride_is_one) and every node of the expression
   * must have a native load for `T` (nda::has_simd_load_v). Expressions that do not qualify are evaluated element by
   * element, e.g. mixed value types (a double array plus a float array), scalars of a wider type (`2.0 * float_array`
   * has value type double), functors without a `load` member and strided fastest dimensions.
   *
   * @code{.cpp}
   * nda::array<double, 2> a(8, 8), b(8, 8);
   * static_assert(nda::is_simd_enabled_v<decltype(2.0 * a + b)>);                  // arithmetic tree, broadcast scalar
   * static_assert(not nda::is_simd_enabled_v<decltype(a(nda::range(0, 4), 0))>); // strided fastest dimension
   * @endcode
   *
   * @tparam A Type of the expression.
   * @tparam T Scalar type the expression has to produce (defaults to its own value type).
   */
  template <typename A, typename T = get_value_t<A>>
  inline constexpr bool is_simd_enabled_v = has_layout_smallest_stride_is_one<A> and has_simd_load_v<A, std::remove_cvref_t<T>>;

  /** @} */

} // namespace nda
