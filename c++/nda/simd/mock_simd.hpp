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
#include <tuple>
#include <utility>

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
   * `load(native_simd<T>...)`, which stores the argument batches to arrays, applies the scalar call to every lane and
   * loads the results back. All arguments and the result must have the same type `T`. A mapped function with such a
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

    template <typename... Args>
    FORCEINLINE value_t operator()(Args &&...args) const {
      static_assert((std::is_same_v<value_t, std::remove_cvref_t<Args>> and ...), "All types have to be the same.");
      return static_cast<const Derived *>(this)->operator()(std::forward<Args>(args)...);
    }

    private:
    FORCEINLINE std::array<value_t, simd_t::size> convert_simd_to_array(const simd_t &a) const {
      alignas(simd_t::arch_type::alignment()) std::array<T, simd_t::size> result;
      a.store_aligned(result.data());
      return result;
    }

    template <size_t... Is, typename... Args>
    FORCEINLINE auto make_array_tuple(std::index_sequence<Is...>, const std::tuple<Args...> &args_tuple) const {
      return std::make_tuple(convert_simd_to_array(std::get<Is>(args_tuple))...);
    }

    template <size_t... Is, typename... Args>
    FORCEINLINE auto apply_function(std::index_sequence<Is...>, const std::tuple<Args...> &array_tuple) const {
      auto build = [&]<size_t... Js>(std::index_sequence<Js...>) {
        auto compute_one_element = [&](size_t i) -> T { return static_cast<const Derived *>(this)->operator()(std::get<Is>(array_tuple)[i]...); };

        alignas(simd_t::arch_type::alignment()) std::array<T, simd_t::size> result_array = {compute_one_element(Js)...};

        return result_array;
      };

      return build(std::make_index_sequence<simd_t::size>{});
    }

    public:
    template <typename... Args>
    [[nodiscard]] FORCEINLINE simd_t load(Args &&...args) const {
      static_assert((std::is_same_v<simd_t, std::remove_cvref_t<Args>> and ...), "All types have to be the same.");
      constexpr size_t args_size        = sizeof...(Args);
      std::tuple<Args &&...> args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
      auto array_tuple                  = make_array_tuple(std::make_index_sequence<args_size>{}, args_tuple);
      alignas(simd_t::arch_type::alignment()) std::array<value_t, simd_t::size> result_array;
      result_array = apply_function(std::make_index_sequence<args_size>{}, array_tuple);
      return simd_t::load_aligned(result_array.data());
    }
  };

  /** @} */

} // namespace nda::simd

#endif // NDA_HAVE_XSIMD
