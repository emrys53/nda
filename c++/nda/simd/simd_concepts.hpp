// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides concepts for SIMD batches and callables with native SIMD loads.
 */

#pragma once

#include "../concepts.hpp"
#include "./simd.hpp"

#include <concepts>
#include <type_traits>

namespace nda {

  /**
   * @addtogroup simd
   * @{
   */

#ifdef NDA_HAVE_XSIMD
  /**
   * @brief Check if xsimd has a native batch for a given scalar type.
   * @tparam S Type to check.
   */
  template <typename S>
  concept Vectorizable = xsimd::has_simd_register<std::remove_cvref_t<S>>::value;

  /// Check if native SIMD batches of the result and all argument types have the same number of lanes.
  template <typename Out, typename... In>
  concept SimdPreservesLaneWidth = Vectorizable<Out> and (Vectorizable<In> and ...)
     and ((simd::native_simd<std::remove_cvref_t<Out>>::size == simd::native_simd<std::remove_cvref_t<In>>::size) and ...);

  /**
   * @brief Check if `F::load` maps native SIMD batches of `In...` to a native SIMD batch of `Out`.
   *
   * @details The types may differ, e.g. `abs` maps a complex batch to a real batch. `Out` must also be the result
   * type of the scalar call `f(in...)`, so that the SIMD and the scalar path produce the same type.
   *
   * @tparam F Callable type.
   * @tparam Out Scalar result type.
   * @tparam In Scalar argument types.
   */
  template <typename F, typename Out, typename... In>
  concept LoadWithNativeSimdTo = Vectorizable<Out> and (Vectorizable<In> and ...)
     and requires(F const &f, simd::native_simd<In> const &...xs) {
           { f.load(xs...) } -> std::same_as<simd::native_simd<Out>>;
         }
     // the batch result must match the scalar result; checked last so that unsupported scalar bodies are never instantiated
     and std::same_as<std::remove_cvref_t<std::invoke_result_t<F const &, In const &...>>, std::remove_cvref_t<Out>>;
#else
  // Without xsimd no type is vectorizable, so all SIMD dispatch degrades to the scalar path.
  template <typename S>
  concept Vectorizable = false;

  template <typename Out, typename... In>
  concept SimdPreservesLaneWidth = false;

  template <typename F, typename Out, typename... In>
  concept LoadWithNativeSimdTo = false;
#endif

  /// Check if a scalar type is arithmetic and has a native SIMD batch.
  template <typename T>
  concept SimdArithmetic = Vectorizable<T> and std::is_arithmetic_v<std::remove_cvref_t<T>>;

  /// Check if an arithmetic scalar type is signed and has a native SIMD batch.
  template <typename T>
  concept SimdSigned = SimdArithmetic<T> and std::is_signed_v<std::remove_cvref_t<T>>;

  /// Check if a scalar type is float or double and has a native SIMD batch.
  template <typename T>
  concept SimdReal = Vectorizable<T> and FloatOrDouble<T>;

  /// Check if a scalar type is complex float or complex double and has a native SIMD batch.
  template <typename T>
  concept SimdComplex = Vectorizable<T> and is_complex_v<T> and SimdReal<remove_complex_t<T>>;

  /// Check if a scalar type is float, double or their complex counterparts and has a native SIMD batch.
  template <typename T>
  concept SimdRealOrComplex = SimdReal<T> or SimdComplex<T>;

  /** @} */

} // namespace nda
