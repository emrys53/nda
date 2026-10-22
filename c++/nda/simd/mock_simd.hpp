// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a CRTP base that gives a scalar functor a lane-by-lane SIMD load (requires xsimd).
 */

#pragma once

#include "../macros.hpp"
#include "./simd_concepts.hpp"

#include <array>
#include <cstddef>

#ifdef NDA_HAVE_XSIMD

namespace nda::simd {

  /**
   * @addtogroup simd
   * @{
   */

  /**
   * @brief CRTP base that gives a scalar functor a SIMD `load` by applying the scalar call lane by lane.
   *
   * @details A functor `F` deriving from `mock_simd<F, T>` and defining `T operator()(T...) const` gains
   * `load(native_simd<U>...)`, which stores the argument batches to arrays, applies the scalar call to every lane (with
   * the usual conversions of a scalar call) and loads the results back as a `native_simd<T>`. A mapped function with such a
   * functor takes part in SIMD expression trees (see nda::simd::has_load_function) without a hand-written batch
   * implementation; it is meant for tests and prototyping, since it is not faster than the scalar path.
   *
   * @code{.cpp}
   * struct plus_one : nda::simd::mock_simd<plus_one, double> {
   *   double operator()(double x) const { return x + 1; }
   * };
   * nda::array<double, 1> a(16);
   * static_assert(nda::is_simd_enabled_v<decltype(nda::map(plus_one{})(a))>);
   * @endcode
   *
   * @tparam Derived Functor type (CRTP).
   * @tparam T Scalar type of the arguments and of the result.
   */
  template <typename Derived, Vectorizable T>
  struct mock_simd {
    using value_t = T;
    using simd_t  = native_simd<T>;

    /**
     * @brief Apply the scalar call of the derived functor to every lane of the argument batches.
     *
     * @tparam Args Types of the argument batches (nda::simd::NativeSimd); the scalar call converts their lanes.
     * @param args Argument batches, one per argument of the scalar call.
     * @return `simd_t` batch of the results.
     */
    template <NativeSimd... Args>
    [[nodiscard]] FORCEINLINE simd_t load(Args const &...args) const {
      // store the argument batches to arrays, apply the scalar call lane by lane and load the results back
      auto const to_array = []<NativeSimd B>(B const &b) {
        alignas(B::arch_type::alignment()) std::array<typename B::value_type, B::size> a;
        b.store_aligned(a.data());
        return a;
      };
      return [this](auto const &...in) {
        alignas(simd_t::arch_type::alignment()) std::array<value_t, simd_t::size> out;
        for (std::size_t i = 0; i < simd_t::size; ++i) out[i] = static_cast<Derived const &>(*this)(in[i]...);
        return simd_t::load_aligned(out.data());
      }(to_array(args)...);
    }
  };

  /** @} */

} // namespace nda::simd

#endif // NDA_HAVE_XSIMD
