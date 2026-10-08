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
#include <type_traits>
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

    // Preserve floating-point precision; retain double results for integral inputs.
    template <std::floating_point T>
    T abs2(T x) { return x * x; }

    template <std::integral T>
    double abs2(T x) {
      auto const y = static_cast<double>(x);
      return y * y;
    }

    template <std::floating_point T>
    T abs2(std::complex<T> z) { return std::norm(z); }

    template <std::integral T>
    double abs2(std::complex<T> z) {
      std::complex<double> const y(static_cast<double>(z.real()), static_cast<double>(z.imag()));
      return std::norm(y);
    }

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
        requires SimdPreservesLaneWidth<remove_complex_t<T>, T>
      FORCEINLINE native_simd<remove_complex_t<T>> load(native_simd<T> const &x) const {
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

    template <typename S = double>
    struct reciprocal_f {
      S alpha{1};
      FORCEINLINE auto operator()(auto const &x) const {
        if constexpr (Scalar<std::remove_cvref_t<decltype(x)>>) {
          return alpha / x;
        } else {
          return alpha * reciprocal(x); // nested arrays: nda::reciprocal, found by ADL at instantiation
        }
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires(PreservesScalarType<reciprocal_f, T>)
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return native_simd<T>(static_cast<T>(alpha)) / x;
      }
#endif
    };

    template <typename P = double>
    struct pow_f {
      P exponent;
      FORCEINLINE auto operator()(auto const &x) const {
        using std::pow;
        return pow(x, exponent);
      }
#ifdef NDA_HAVE_XSIMD
      // Integer exponents on complex inputs use a distinct std::pow overload; keep those on the scalar path.
      template <SimdRealOrComplex T>
        requires(PreservesScalarType<pow_f, T>
                 and (std::same_as<P, remove_complex_t<T>> or std::same_as<P, T>
                      or (std::same_as<T, double> and std::is_arithmetic_v<P>)))
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
   *         The exponent retains its type, so the scalar result follows `std::pow` overload resolution.
   */
  template <ArrayOrScalar A, Scalar P>
  auto pow(A &&a, P p) {
    return nda::map(detail::pow_f<P>{p})(std::forward<A>(a));
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
   * @return A lazy nda::expr_call object (nda::Array) or the result of \f$ 1 / x \f$ applied to the object
   * (nda::Scalar), in the input's floating-point precision. Integral inputs return double.
   */
  template <ArrayOrScalar A>
  auto reciprocal(A &&a) {
    if constexpr (Array<get_value_t<A>>) {
      return nda::map([](auto const &x) { return nda::reciprocal(x); })(std::forward<A>(a));
    } else {
      using real_t = remove_complex_t<get_value_t<A>>;
      using numerator_t = std::conditional_t<std::is_floating_point_v<real_t>, real_t, double>;
      return nda::map(detail::reciprocal_f<numerator_t>{numerator_t{1}})(std::forward<A>(a));
    }
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
