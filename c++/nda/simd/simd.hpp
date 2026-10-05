// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once

#ifdef NDA_HAVE_XSIMD

#include <xsimd/xsimd.hpp>

namespace nda {

  /// Native SIMD batch of the plain scalar type `T` (transparent alias, so `T` is deducible from a batch argument).
  template <typename T>
  using native_simd = xsimd::batch<T>;

  /// SIMD batch of the plain scalar type `T` with exactly `Width` lanes.
  template <typename T, size_t Width>
  using fixed_size_simd = xsimd::make_sized_batch_t<T, Width>;

} // namespace nda

#endif // NDA_HAVE_XSIMD
