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

#include <type_traits>

namespace nda {

  // forward declarations of the node types the traits below dispatch on
  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  class basic_array;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  class basic_array_view;

  template <char OP, Array A>
  struct expr_unary;

  template <char OP, ArrayOrScalar L, ArrayOrScalar R>
  struct expr;

  template <typename F, Array... A>
  struct expr_call;

  /**
   * @addtogroup utils_concepts
   * @{
   */

  /// Check if a given type is an nda::basic_array.
  template <typename T>
  concept IsBasicArray = requires {
    []<typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>(
       basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>) {}(std::declval<T>());
  };

  /// Check if a given type is an nda::basic_array_view.
  template <typename T>
  concept IsBasicArrayView = requires {
    []<typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>(
       basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>) {}(std::declval<T>());
  };

  /// Check if a given type is an nda::expr_unary.
  template <typename T>
  concept IsExprUnary = requires { []<char OP, Array A>(expr_unary<OP, A>) {}(std::declval<T>()); };

  /// Check if a given type is an nda::expr_call.
  template <typename T>
  concept IsExprCall = requires { []<typename F, Array... As>(expr_call<F, As...>) {}(std::declval<T>()); };

  /// Check if a given type is an nda::expr.
  template <typename T>
  concept IsExpr = requires { []<char OP, typename L, typename R>(expr<OP, L, R>) {}(std::declval<T>()); };

  /** @} */

  namespace simd {

    /**
     * @addtogroup utils_type_traits
     * @{
     */

    /**
     * @brief Check if an expression tree can be loaded as a `native_simd<T_in>` batch using only native SIMD loads.
     *
     * @details Every node must produce the value type `T` it is asked for:
     * - arrays/views: `T` must be nda::Vectorizable,
     * - nda::expr_unary, nda::expr: the operands are checked with the same `T` (a scalar operand is broadcast as `T` and
     *   must be convertible to it),
     * - nda::expr_call: may change the value type (e.g. `abs` of a complex array); the arguments are checked with their
     *   own value types, all batches must have the same width (nda::SimdPreservesLaneWidth) and `F::load` must map
     *   the argument batches to a batch of `T` (nda::LoadWithNativeSimdTo).
     *
     * @tparam A_in Type of the expression.
     * @tparam T_in Scalar type the expression has to produce (defaults to its own value type).
     * @return True if the whole tree has native SIMD loads.
     */
    template <typename A_in, typename T_in = get_value_t<A_in>>
    static consteval bool has_load_function() {
      using A         = std::remove_cvref_t<A_in>;
      using T         = std::remove_cvref_t<T_in>;
      using ValueType = get_value_t<A>;
      if constexpr (not std::is_same_v<ValueType, T>) { return false; }
      if constexpr (IsBasicArray<A> or IsBasicArrayView<A>) { return Vectorizable<ValueType>; }
      if constexpr (IsExprUnary<A>) {
        return []<char OP, Array E>(std::type_identity<expr_unary<OP, E>>) { return has_load_function<E, T>(); }(std::type_identity<A>{});
      }
      if constexpr (IsExprCall<A>) {
        return []<typename F, Array... As>(std::type_identity<expr_call<F, As...>>) {
          // check the widths before probing F::load, which instantiates its signature
          if constexpr (not SimdPreservesLaneWidth<ValueType, get_value_t<As>...>) {
            return false;
          } else {
            return LoadWithNativeSimdTo<F, ValueType, get_value_t<As>...> and (has_load_function<As, get_value_t<As>>() and ...);
          }
        }(std::type_identity<A>{});
      }
      if constexpr (IsExpr<A>) {
        return []<char OP, typename L, typename R>(std::type_identity<expr<OP, L, R>>) {
          if constexpr (is_scalar_v<L>) { return has_load_function<R, T>() and std::is_convertible_v<std::remove_cvref_t<L>, T>; }
          if constexpr (is_scalar_v<R>) { return has_load_function<L, T>() and std::is_convertible_v<std::remove_cvref_t<R>, T>; }
          return has_load_function<L, T>() and has_load_function<R, T>();
        }(std::type_identity<A>{});
      }
      return false;
    }

    /** @} */

  } // namespace simd

  /**
   * @brief Constexpr variable that is true if an expression of type `A` is evaluated into values of type `T` with
   * native SIMD loads.
   *
   * @details nda::simd::has_load_function is only probed for a contiguous layout, since probing instantiates `load`
   * signatures that may not compile.
   *
   * @tparam A Type of the expression.
   * @tparam T Scalar type the expression has to produce (defaults to its own value type).
   */
  template <typename A, typename T = get_value_t<A>>
  inline constexpr bool is_simd_enabled_v = [] {
    if constexpr (has_contiguous_layout<A>) {
      return simd::has_load_function<A, T>();
    } else {
      return false;
    }
  }();

} // namespace nda
