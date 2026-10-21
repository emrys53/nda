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
#include "./simd/simd_concepts.hpp"
#include "./traits.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
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

    // Functors of the mapped functions below. operator() is the scalar call; with xsimd, load() is the batch version,
    // constrained to the scalar domain xsimd supports (see the Simd* concepts). nda::map applies operator() element
    // by element and, if the functor has a load for the argument batches, load() in SIMD blocks (see
    // nda::simd::has_load_function); a functor without a matching load keeps the expression on the scalar path.
    struct conj_f {
      FORCEINLINE auto operator()(auto const &x) const { return conj(x); }

#ifdef NDA_HAVE_XSIMD
      template <typename T>
        requires(SimdArithmetic<T> or SimdRealOrComplex<T>)
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
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
      FORCEINLINE simd::native_simd<remove_complex_t<T>> load(simd::native_simd<T> const &x) const {
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
      FORCEINLINE simd::native_simd<remove_complex_t<T>> load(simd::native_simd<T> const &x) const {
        if constexpr (is_complex_v<T>) {
          using xsimd::imag;
          return imag(x);
        } else {
          return simd::native_simd<T>(0);
        }
      }
#endif
    };

    struct abs2_f {
      FORCEINLINE auto operator()(auto const &x) const { return detail::abs2(x); }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires SimdPreservesLaneWidth<remove_complex_t<T>, T>
      FORCEINLINE simd::native_simd<remove_complex_t<T>> load(simd::native_simd<T> const &x) const {
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
        if constexpr (std::is_unsigned_v<std::remove_cvref_t<decltype(x)>>) {
          return x;
        } else {
          using std::abs;
          return abs(x);
        }
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdSigned T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::abs(x);
      }

      template <SimdComplex T>
        requires SimdPreservesLaneWidth<remove_complex_t<T>, T>
      FORCEINLINE simd::native_simd<remove_complex_t<T>> load(simd::native_simd<T> const &x) const {
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
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
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
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return simd::native_simd<T>(static_cast<T>(alpha)) / x;
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
        requires(std::same_as<P, remove_complex_t<T>> or std::same_as<P, T> or (std::same_as<T, double> and std::is_arithmetic_v<P>))
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        using xsimd::pow;
        return pow(x, simd::native_simd<T>(exponent));
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
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x, simd::native_simd<T> const &y) const {
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
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x, simd::native_simd<T> const &y) const {
        using xsimd::min;
        return min(x, y);
      }
#endif
    };

    struct mul_f {
      FORCEINLINE auto operator()(auto const &x, auto const &y) const { return x * y; }

#ifdef NDA_HAVE_XSIMD
      template <Vectorizable T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x, simd::native_simd<T> const &y) const {
        return x * y;
      }
#endif
    };

    struct exp_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::exp;
        return exp(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::exp(x);
      }
#endif
    };

    struct cos_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::cos;
        return cos(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::cos(x);
      }
#endif
    };

    struct sin_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::sin;
        return sin(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::sin(x);
      }
#endif
    };

    struct tan_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::tan;
        return tan(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::tan(x);
      }
#endif
    };

    struct cosh_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::cosh;
        return cosh(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::cosh(x);
      }
#endif
    };

    struct sinh_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::sinh;
        return sinh(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::sinh(x);
      }
#endif
    };

    struct tanh_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::tanh;
        return tanh(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::tanh(x);
      }
#endif
    };

    struct acos_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::acos;
        return acos(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::acos(x);
      }
#endif
    };

    struct asin_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::asin;
        return asin(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::asin(x);
      }
#endif
    };

    struct atan_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::atan;
        return atan(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::atan(x);
      }
#endif
    };

    struct log_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::log;
        return log(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::log(x);
      }
#endif
    };

    struct sqrt_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::sqrt;
        return sqrt(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
      FORCEINLINE simd::native_simd<T> load(simd::native_simd<T> const &x) const {
        return xsimd::sqrt(x);
      }
#endif
    };

  } // namespace detail

  // The scalar overloads of the <cmath> family are the std ones, brought in by using-declarations rather than
  // nda function templates: libc++ declares generic templates like pow(A1, A2) in the global namespace, which would
  // tie with such templates for unqualified calls under `using namespace nda`. The using-declarations must follow the
  // <cmath>, <complex> and <cstdlib> includes, since they only capture overloads declared before them.
#ifdef NDA_DOXYGEN
  /**
   * @brief Function pow for nda::Scalar types (uses `std::pow`).
   */
  template <Scalar S, Scalar P>
  auto pow(S s, P p);
#endif
  using std::pow;

  /**
   * @brief Function pow for nda::Array types (lazy and coefficient-wise).
   *
   * @tparam A nda::Array type.
   * @tparam S nda::Scalar exponent type.
   * @param a nda::Array object.
   * @param p Exponent value.
   * @return A lazy nda::expr_call object.
   */
  template <Array A, Scalar S>
  auto pow(A &&a, S p) {
    return nda::map(detail::pow_f<S>{p})(std::forward<A>(a));
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

#ifdef NDA_DOXYGEN
  /// \brief Function abs for nda::Scalar types (uses `std::abs`).
  template <Scalar S>
  auto abs(S s);
#endif
  using std::abs;

  /// \brief Function abs for nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto abs(A &&a) {
    return nda::map(detail::abs_f{})(std::forward<A>(a));
  }

  /// \brief Function imag for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::imag` applied to the object (nda::Scalar).
  template <ArrayOrScalar A>
  auto imag(A &&a) {
    return nda::map(detail::imag_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function floor for nda::Scalar types (uses `std::floor`).
  template <Scalar S>
  auto floor(S s);
#endif
  using std::floor;

  /// \brief Function floor for nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto floor(A &&a) {
    return nda::map(detail::floor_f{})(std::forward<A>(a));
  }

  /// \brief Function real for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type..
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `nda::detail::real` applied to the object (nda::Scalar).
  template <ArrayOrScalar A>
  auto real(A &&a) {
    return nda::map(detail::real_f{})(std::forward<A>(a));
  }

  /// \brief Function abs2 for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type..
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `nda::detail::abs2` applied to the object (nda::Scalar).
  template <ArrayOrScalar A>
  auto abs2(A &&a) {
    return nda::map(detail::abs2_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function isnan for real nda::Scalar types (uses `std::isnan`).
  template <Scalar S>
    requires(not is_complex_v<S>)
  auto isnan(S s);
#endif
  using std::isnan;

  /// \brief Function isnan for nda::Array types and complex nda::Scalar types (lazy and coefficient-wise for nda::Array
  /// types).
  ///
  /// \details A complex scalar is NaN if either component is NaN.
  ///
  /// \tparam A nda::Array or complex nda::Scalar type.
  /// \param a nda::Array or complex nda::Scalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `nda::detail::isnan` applied to the object
  /// (complex nda::Scalar).
  template <typename A>
    requires(Array<A> or is_complex_v<A>)
  auto isnan(A &&a) {
    return nda::map([](auto const &x) { return detail::isnan(x); })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function exp for nda::Scalar types (uses `std::exp`).
  template <Scalar S>
  auto exp(S s);
#endif
  using std::exp;

  /// \brief Function exp for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto exp(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::exp_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function cos for nda::Scalar types (uses `std::cos`).
  template <Scalar S>
  auto cos(S s);
#endif
  using std::cos;

  /// \brief Function cos for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto cos(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::cos_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function sin for nda::Scalar types (uses `std::sin`).
  template <Scalar S>
  auto sin(S s);
#endif
  using std::sin;

  /// \brief Function sin for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto sin(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::sin_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function tan for nda::Scalar types (uses `std::tan`).
  template <Scalar S>
  auto tan(S s);
#endif
  using std::tan;

  /// \brief Function tan for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto tan(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::tan_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function cosh for nda::Scalar types (uses `std::cosh`).
  template <Scalar S>
  auto cosh(S s);
#endif
  using std::cosh;

  /// \brief Function cosh for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto cosh(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::cosh_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function sinh for nda::Scalar types (uses `std::sinh`).
  template <Scalar S>
  auto sinh(S s);
#endif
  using std::sinh;

  /// \brief Function sinh for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto sinh(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::sinh_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function tanh for nda::Scalar types (uses `std::tanh`).
  template <Scalar S>
  auto tanh(S s);
#endif
  using std::tanh;

  /// \brief Function tanh for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto tanh(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::tanh_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function acos for nda::Scalar types (uses `std::acos`).
  template <Scalar S>
  auto acos(S s);
#endif
  using std::acos;

  /// \brief Function acos for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto acos(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::acos_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function asin for nda::Scalar types (uses `std::asin`).
  template <Scalar S>
  auto asin(S s);
#endif
  using std::asin;

  /// \brief Function asin for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto asin(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::asin_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function atan for nda::Scalar types (uses `std::atan`).
  template <Scalar S>
  auto atan(S s);
#endif
  using std::atan;

  /// \brief Function atan for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto atan(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::atan_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function log for nda::Scalar types (uses `std::log`).
  template <Scalar S>
  auto log(S s);
#endif
  using std::log;

  /// \brief Function log for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto log(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::log_f{})(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function sqrt for nda::Scalar types (uses `std::sqrt`).
  template <Scalar S>
  auto sqrt(S s);
#endif
  using std::sqrt;

  /// \brief Function sqrt for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto sqrt(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::sqrt_f{})(std::forward<A>(a));
  }

  /** @} */

} // namespace nda
