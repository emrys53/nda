// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides some custom implementations of standard mathematical functions used for lazy, coefficient-wise array
 * operations.
 */

#pragma once

#include "./concepts.hpp"
#include "./map.hpp"
#include "./traits.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <utility>

namespace nda {

  /**
   * @addtogroup av_math
   * @{
   */

  namespace detail {

    // Get the real part of a scalar.
    template <nda::Scalar S>
    auto real(S x) {
      if constexpr (is_complex_v<S>) {
        return std::real(x);
      } else {
        return x;
      }
    }

    // Get the imaginary part of a scalar.
    template <nda::Scalar S>
    auto imag(S x) {
      return std::imag(x);
    }

    // Get the complex conjugate of a scalar.
    template <nda::Scalar S>
    auto conj(S x) {
      if constexpr (is_complex_v<S>) {
        return std::conj(x);
      } else {
        return x;
      }
    }

    // Get the squared absolute value of a double.
    inline double abs2(double x) { return x * x; }

    // Get the squared absolute value of a std::complex<double>.
    inline double abs2(std::complex<double> z) { return (conj(z) * z).real(); }

    // Check if a std::complex<double> is NaN.
    inline bool isnan(std::complex<double> const &z) { return std::isnan(z.real()) or std::isnan(z.imag()); }

    // Check if a real floating-point scalar is NaN.
    template <std::floating_point S>
    bool isnan(S x) {
      return std::isnan(x);
    }

    struct conj_f {
      FORCEINLINE auto operator()(auto const &x) const { return conj(x); }

#ifdef NDA_HAVE_XSIMD
      template <typename T>
        requires((SimdArithmetic<T> or SimdRealOrComplex<T>) and PreservesScalarType<conj_f, T>)
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        if constexpr (is_complex_v<T>) {
          using xsimd::conj;
          return conj(x);
        } else {
          return x;
        }
      }
#endif
    };

    struct real_f {
      FORCEINLINE auto operator()(auto const &x) const { return detail::real(x); }
#ifdef NDA_HAVE_XSIMD
      template <typename T>
        requires((SimdArithmetic<T> or SimdRealOrComplex<T>) and SimdPreservesLaneWidth<remove_complex_t<T>, T>)
      FORCEINLINE native_simd<remove_complex_t<T>> load(native_simd<T> const &x) const {
        if constexpr (is_complex_v<T>) {
          using xsimd::real;
          return real(x);
        } else {
          return x;
        }
      }
#endif
    };

    struct imag_f {
      FORCEINLINE auto operator()(auto const &x) const { return detail::imag(x); }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires SimdPreservesLaneWidth<remove_complex_t<T>, T>
      FORCEINLINE native_simd<remove_complex_t<T>> load(native_simd<T> const &x) const {
        if constexpr (is_complex_v<T>) {
          using xsimd::imag;
          return imag(x);
        } else {
          return native_simd<T>(0);
        }
      }
#endif
    };

    struct abs2_f {
      FORCEINLINE auto operator()(auto const &x) const { return detail::abs2(x); }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires(std::same_as<remove_complex_t<T>, double> and SimdPreservesLaneWidth<double, T>)
      FORCEINLINE native_simd<double> load(native_simd<T> const &x) const {
        if constexpr (is_complex_v<T>) {
          using xsimd::norm;
          return norm(x);
        } else {
          return x * x;
        }
      }
#endif
    };

    struct abs_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::abs;
        return abs(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdSigned T>
        requires PreservesScalarType<abs_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::abs(x);
      }

      template <SimdComplex T>
        requires SimdPreservesLaneWidth<remove_complex_t<T>, T>
      FORCEINLINE native_simd<remove_complex_t<T>> load(native_simd<T> const &x) const {
        using xsimd::abs;
        return abs(x);
      }
#endif
    };

    struct floor_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::floor;
        return floor(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdReal T>
        requires PreservesScalarType<floor_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::floor(x);
      }
#endif
    };

    struct reciprocal_f {
      FORCEINLINE auto operator()(auto const &x) const {
        if constexpr (Scalar<std::remove_cvref_t<decltype(x)>>) {
          return 1.0 / x;
        } else {
          return reciprocal(x); // nested arrays: nda::reciprocal, found by ADL at instantiation
        }
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires(std::same_as<remove_complex_t<T>, double> and PreservesScalarType<reciprocal_f, T>)
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return native_simd<T>(1) / x;
      }
#endif
    };

    struct pow_f {
      double exponent;
      FORCEINLINE auto operator()(auto const &x) const {
        using std::pow;
        return pow(x, exponent);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires(std::same_as<remove_complex_t<T>, double> and PreservesScalarType<pow_f, T>)
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        using xsimd::pow;
        return pow(x, native_simd<T>(exponent));
      }
#endif
    };

    struct max_f {
      FORCEINLINE auto operator()(auto const &x, auto const &y) const {
        using std::max;
        return max(x, y);
      }

#ifdef NDA_HAVE_XSIMD
      template <SimdArithmetic T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x, native_simd<T> const &y) const {
        using xsimd::max;
        return max(x, y);
      }
#endif
    };

    struct min_f {
      FORCEINLINE auto operator()(auto const &x, auto const &y) const {
        using std::min;
        return min(x, y);
      }

#ifdef NDA_HAVE_XSIMD
      template <SimdArithmetic T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x, native_simd<T> const &y) const {
        using xsimd::min;
        return min(x, y);
      }
#endif
    };

    struct mul_f {
      FORCEINLINE auto operator()(auto const &x, auto const &y) const { return x * y; }

#ifdef NDA_HAVE_XSIMD
      template <Vectorizable T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x, native_simd<T> const &y) const {
        return x * y;
      }
#endif
    };
  } // namespace detail

  /**
   * @brief Function pow for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
   *
   * @tparam A nda::ArrayOrScalar type.
   * @param a nda::ArrayOrScalar object.
   * @param p Exponent value.
   * @return A lazy nda::expr_call object (nda::Array) or the result of `std::pow` applied to the object (nda::Scalar).
   */
  template <ArrayOrScalar A>
  auto pow(A &&a, double p) {
    return nda::map(detail::pow_f{p})(std::forward<A>(a));
  }

  /**
   * @brief Function conj for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types with a complex
   * value type).
   *
   * @tparam A nda::ArrayOrScalar type.
   * @param a nda::ArrayOrScalar object.
   * @return A lazy nda::expr_call object (nda::Array and complex valued), the forwarded input object (nda::Array and
   * not complex valued) or the complex conjugate of the scalar input.
   */
  template <ArrayOrScalar A>
  decltype(auto) conj(A &&a) {
    if constexpr (is_complex_v<get_value_t<A>>) {
      return nda::map(detail::conj_f{})(std::forward<A>(a));
    } else {
      return std::forward<A>(a);
    }
  }

  /**
   * @brief Reciprocal function for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
   * 
   * @tparam A nda::ArrayOrScalar type.
   * @param a nda::ArrayOrScalar object.
   * @return A lazy nda::expr_call object (nda::Array) or the result of \f$ 1.0 / x \f$ applied to the object 
   * (nda::Scalar).
   */
  template <ArrayOrScalar A>
  auto reciprocal(A &&a) {
    return nda::map(detail::reciprocal_f{})(std::forward<A>(a));
  }

  /**
   * @brief Function max for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
   *
   * @tparam A nda::ArrayOrScalar type.
   * @tparam B nda::ArrayOrScalar type.
   * @param a First operand.
   * @param b Second operand.
   * @return A lazy nda::expr_call object (nda::Array) or the result of `std::max` applied to the inputs (nda::Scalar).
   */
  template <ArrayOrScalar A, ArrayOrScalar B>
    requires(((Scalar<A> and Scalar<B>) or (Array<A> and Array<B> and get_rank<A> == get_rank<B>))
             and not is_complex_v<get_value_t<A>> and not is_complex_v<get_value_t<B>>)
  [[nodiscard]] auto max(A &&a, B &&b) {
    return nda::map(detail::max_f{})(std::forward<A>(a), std::forward<B>(b));
  }

  /**
   * @brief Function min for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
   *
   * @tparam A nda::ArrayOrScalar type.
   * @tparam B nda::ArrayOrScalar type.
   * @param a First operand.
   * @param b Second operand.
   * @return A lazy nda::expr_call object (nda::Array) or the result of `std::min` applied to the inputs (nda::Scalar).
   */
  template <ArrayOrScalar A, ArrayOrScalar B>
    requires(((Scalar<A> and Scalar<B>) or (Array<A> and Array<B> and get_rank<A> == get_rank<B>))
             and not is_complex_v<get_value_t<A>> and not is_complex_v<get_value_t<B>>)
  [[nodiscard]] auto min(A &&a, B &&b) {
    return nda::map(detail::min_f{})(std::forward<A>(a), std::forward<B>(b));
  }

  /** @} */

} // namespace nda
