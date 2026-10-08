// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic in-place elementwise binary tensor operation with cuTENSOR/nda dispatch.
 */

#pragma once

#include "./interface/cutensor_interface.hpp"
#include "./tools.hpp"
#include "../algorithms.hpp"
#include "../exceptions.hpp"
#include "../mapped_functions.hpp"
#include "../mapped_functions.hxx"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <string_view>
#include <utility>

namespace nda::tensor {

  /**
   * @addtogroup tensor_ops
   * @{
   */

  namespace detail {
    // Calls f with the lazy expression op(alpha * x, beta * y), so that nda can vectorize its assignment. Every op
    // yields a different expression type, hence the visitor instead of a return value. T is the tensor value type.
    // The scalars stay apart from x and y so that NORM_2 scales by std::norm(alpha) and PROD by alpha * beta once,
    // instead of multiplying every element, which for complex T is a full complex multiplication.
    template <typename T, typename X, typename Y, typename F>
    void visit_binary_expr(binary_op op, T alpha, X const &x, T beta, Y const &y, F &&f) {
      switch (op) {
        case binary_op::SUM: f(alpha * x + beta * y); return;
        case binary_op::PROD: f((alpha * beta) * nda::hadamard(x, y)); return;
        case binary_op::SUM_ABS: f(nda::abs(alpha * x) + nda::abs(beta * y)); return;
        case binary_op::MAX_ABS: f(nda::max(nda::abs(alpha * x), nda::abs(beta * y))); return;
        case binary_op::MIN_ABS: f(nda::min(nda::abs(alpha * x), nda::abs(beta * y))); return;
        // nda::sqrt rejects matrix algebra, so map sqrt_f directly to also accept nda::matrix operands
        case binary_op::NORM_2: f(nda::map(nda::detail::sqrt_f{})(std::norm(alpha) * nda::abs2(x) + std::norm(beta) * nda::abs2(y))); return;
        case binary_op::MAX:
        case binary_op::MIN:
          if constexpr (is_complex_v<T>) {
            NDA_RUNTIME_ERROR << "nda::tensor: binary_op::MAX/MIN are unsupported for complex value types";
          } else if (op == binary_op::MAX) {
            f(nda::max(alpha * x, beta * y));
          } else {
            f(nda::min(alpha * x, beta * y));
          }
          return;
      }
      // No default case, so -Wswitch flags an unhandled binary_op at compile time; this catches out-of-range values.
      NDA_RUNTIME_ERROR << "nda::tensor: unhandled binary_op " << static_cast<int>(op);
    }

    // visit_binary_expr without alpha, for the second trinary operation whose x is the first result. Passing alpha = 1
    // instead would still multiply every element of x by one, since the value is only known at runtime.
    template <typename T, typename X, typename Y, typename F>
    void visit_binary_expr_unscaled(binary_op op, X const &x, T beta, Y const &y, F &&f) {
      switch (op) {
        case binary_op::SUM: f(x + beta * y); return;
        case binary_op::PROD: f(beta * nda::hadamard(x, y)); return;
        case binary_op::SUM_ABS: f(nda::abs(x) + nda::abs(beta * y)); return;
        case binary_op::MAX_ABS: f(nda::max(nda::abs(x), nda::abs(beta * y))); return;
        case binary_op::MIN_ABS: f(nda::min(nda::abs(x), nda::abs(beta * y))); return;
        // nda::sqrt rejects matrix algebra, so map sqrt_f directly to also accept nda::matrix operands
        case binary_op::NORM_2: f(nda::map(nda::detail::sqrt_f{})(nda::abs2(x) + std::norm(beta) * nda::abs2(y))); return;
        case binary_op::MAX:
        case binary_op::MIN:
          if constexpr (is_complex_v<T>) {
            NDA_RUNTIME_ERROR << "nda::tensor: binary_op::MAX/MIN are unsupported for complex value types";
          } else if (op == binary_op::MAX) {
            f(nda::max(x, beta * y));
          } else {
            f(nda::min(x, beta * y));
          }
          return;
      }
      // No default case, so -Wswitch flags an unhandled binary_op at compile time; this catches out-of-range values.
      NDA_RUNTIME_ERROR << "nda::tensor: unhandled binary_op " << static_cast<int>(op);
    }

    // out = op(alpha * a, beta * b)
    template <typename A, typename B, typename Out>
    void assign_binary(binary_op op, get_value_t<A> alpha, A const &a, get_value_t<A> beta, B const &b, Out &&out) { // NOLINT
      visit_binary_expr(op, alpha, a, beta, b, [&](auto const &e) { out = e; });
    }
  } // namespace detail

  /**
   * @brief In-place elementwise binary tensor operation with cuTENSOR/nda dispatch.
   *
   * @details This function performs an in-place elementwise binary operation of the form
   * \f[
   *   B_{\text{idx}_B} \leftarrow \text{op}\bigl(\alpha \, A_{\text{idx}_A}, \, \beta \, B_{\text{idx}_B}\bigr) \;,
   * \f]
   * where \f$ \alpha \f$ and \f$ \beta \f$ are scalars, \f$ A \f$ and \f$ B \f$ are tensors, and \f$ \text{op} \f$ is a
   * binary operation (see nda::tensor::binary_op). The index strings specify how the dimensions of \f$ A \f$ map to
   * those of \f$ B \f$ (Einstein notation); when ranks differ on the cuTENSOR path, indices present in one tensor but
   * absent from the other drive broadcast/reduction on the backend.
   *
   * <details>
   * <summary>**Dispatch order and details**:</summary>
   * - If the input arrays satisfy nda::mem::have_device_compatible_addr_space, cuTENSOR's elementwise binary operation
   * is used.
   * - Otherwise, fallback to a single (vectorizable) lazy nda expression assignment.
   *
   * The supported binary operations depend on the library backend. The nda host fallback requires identical ranks 
   * and identical index strings and supports all nda::tensor::binary_op values.
   * </details>
   *
   * @note \f$ A \f$ is allowed to be a lazy conjugate expression (see nda::blas_lapack::is_conj_array_expr).
   *
   * @tparam A nda::blas_lapack::BlasArrayOrConj type.
   * @tparam B nda::blas_lapack::BlasArrayFor<A> type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param a Input tensor \f$ A \f$.
   * @param idx_a Index string \f$ \text{idx}_A \f$ for tensor \f$ A \f$.
   * @param beta Input scalar \f$ \beta \f$.
   * @param b Input/Output tensor \f$ B \f$.
   * @param idx_b Index string \f$ \text{idx}_B \f$ for tensor \f$ B \f$.
   * @param op Binary operation (default: `binary_op::SUM`).
   */
  template <BlasArrayOrConj A, BlasArrayFor<A> B>
  void elementwise(get_value_t<A> alpha, A const &a, std::string_view idx_a, get_value_t<A> beta, B &&b, std::string_view idx_b, // NOLINT
                   binary_op op = binary_op::SUM) {
    // compile-time checks
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, B>;
    static_assert(!run_on_device || have_cutensor, "nda::tensor::elementwise: cuTENSOR support is required");
    static_assert(run_on_device || get_rank<A> == get_rank<B>, "nda::tensor::elementwise: host fallback requires identical ranks");

    // dispatch to backends
    if constexpr (run_on_device) {
      device::elementwise_binary(alpha, a, idx_a, beta, b, idx_b, b, op);
    } else {
      require_equal_indices(idx_a, idx_b, get_rank<A>, "elementwise");
      detail::assign_binary(op, alpha, a, beta, b, b);
    }
  }

  /// Convenience overload of nda::tensor::elementwise with nda::tensor::default_index strings.
  template <BlasArrayOrConj A, BlasArrayFor<A> B>
  void elementwise(get_value_t<A> alpha, A const &a, get_value_t<A> beta, B &&b, binary_op op = binary_op::SUM) { // NOLINT
    elementwise(alpha, a, default_index<get_rank<A>>(), beta, std::forward<B>(b), default_index<get_rank<B>>(), op);
  }

  /** @} */

} // namespace nda::tensor
