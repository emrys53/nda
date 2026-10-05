// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides concepts for SIMD batches and callables with native SIMD loads.
 */

#pragma once

#include "./simd.hpp"

#include <concepts>
#include <type_traits>

namespace nda {

  /**
   * @addtogroup utils_concepts
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
     and ((native_simd<std::remove_cvref_t<Out>>::size == native_simd<std::remove_cvref_t<In>>::size) and ...);

  namespace simd {
    template <typename Derived, Vectorizable T>
    struct mock_simd;
  }

  /**
   * @brief Check if `F::load` maps native SIMD batches of `In...` to a native SIMD batch of `Out`.
   *
   * @details The types may differ, e.g. `abs` maps a complex batch to a real batch. A callable derived from
   * nda::simd::mock_simd only qualifies for equal types.
   *
   * @tparam F Callable type.
   * @tparam Out Scalar result type.
   * @tparam In Scalar argument types.
   */
  template <typename F, typename Out, typename... In>
  concept LoadWithNativeSimdTo = Vectorizable<Out> and (Vectorizable<In> and ...)
     and (requires(F const &f, native_simd<In> const &...xs) {
           { f.load(xs...) } -> std::same_as<native_simd<Out>>;
         } or (std::is_base_of_v<simd::mock_simd<F, Out>, F> and (std::same_as<std::remove_cvref_t<Out>, std::remove_cvref_t<In>> and ...)));
#else
  // Without xsimd no type is vectorizable, so all SIMD dispatch degrades to the scalar path.
  template <typename S>
  concept Vectorizable = false;

  template <typename Out, typename... In>
  concept SimdPreservesLaneWidth = false;

  template <typename F, typename Out, typename... In>
  concept LoadWithNativeSimdTo = false;
#endif

  /** @} */

} // namespace nda
