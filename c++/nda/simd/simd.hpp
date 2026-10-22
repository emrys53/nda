// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides the SIMD batch type aliases used by nda (only available with xsimd, i.e. `SIMDSupport=ON`).
 */

#pragma once

#ifdef NDA_HAVE_XSIMD

#include <xsimd/xsimd.hpp>

namespace nda::simd {

  /**
   * @addtogroup utils_type_traits
   * @{
   */

  /**
   * @brief Native SIMD batch of the scalar type `T`.
   *
   * @details Transparent alias of `xsimd::batch<T>`, the widest batch the target architecture offers for `T`, e.g.
   * 8 doubles with AVX-512, 4 with AVX2 and 2 with SSE2.
   *
   * @tparam T Scalar type.
   */
  template <typename T>
  using native_simd = xsimd::batch<T>;

  /**
   * @brief SIMD batch of the scalar type `T` with exactly `Width` lanes.
   * @tparam T Scalar type.
   * @tparam Width Number of lanes, which has to be a batch width the target architecture supports for `T`.
   */
  template <typename T, size_t Width>
  using fixed_size_simd = xsimd::make_sized_batch_t<T, Width>;

  /** @} */

} // namespace nda::simd

#endif // NDA_HAVE_XSIMD
