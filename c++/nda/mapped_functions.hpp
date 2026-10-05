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
#include <cstdlib>
#include <concepts>
#include <utility>

namespace nda {

  // Scalar math uses the standard overload sets; NDA overloads below operate on arrays.
  using std::pow;
  using std::abs;
  using std::floor;
  using std::isnan;
  using std::exp;
  using std::cos;
  using std::sin;
  using std::tan;
  using std::cosh;
  using std::sinh;
  using std::tanh;
  using std::acos;
  using std::asin;
  using std::atan;
  using std::log;
  using std::sqrt;


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
        if constexpr (std::is_unsigned_v<std::remove_cvref_t<decltype(x)>>) {
          return x;
        } else {
          using std::abs;
          return abs(x);
        }
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


  } // namespace detail

  /**
   * @brief Function pow for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
   *
   * @tparam A nda::ArrayOrScalar type.
   * @param a nda::ArrayOrScalar object.
   * @param p Exponent value.
   * @return A lazy nda::expr_call object (nda::Array) or the result of `std::pow` applied to the object (nda::Scalar).
   */
  template <Array A>
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


  /// \brief Function abs for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::abs` applied to the object (nda::Scalar).
  template <Array A>
  auto abs(A &&a) {
    return nda::map(detail::abs_f{})(std::forward<A>(a));
  }

  /// \brief Function imag for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of the scalar functor applied to the object (nda::Scalar).
  template <ArrayOrScalar A>
  auto imag(A &&a) {
    return nda::map(detail::imag_f{})(std::forward<A>(a));
  }

  /// \brief Function floor for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of the scalar functor applied to the object (nda::Scalar).
  template <Array A>
  auto floor(A &&a) {
    return nda::map(detail::floor_f{})(std::forward<A>(a));
  }

  /// \brief Function real for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of the scalar functor applied to the object (nda::Scalar).
  template <ArrayOrScalar A>
  auto real(A &&a) {
    return nda::map(detail::real_f{})(std::forward<A>(a));
  }

  /// \brief Function abs2 for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of the scalar functor applied to the object (nda::Scalar).
  template <ArrayOrScalar A>
  auto abs2(A &&a) {
    return nda::map(detail::abs2_f{})(std::forward<A>(a));
  }

  /// \brief Function isnan for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the scalar NaN test (nda::Scalar).
  template <typename A>
    requires(Array<A> or is_complex_v<A>)
  auto isnan(A &&a) {
    return nda::map([](auto const &x) { return detail::isnan(x); })(std::forward<A>(a));
  }

  namespace detail {
    struct exp_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::exp;
        return exp(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires PreservesScalarType<exp_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::exp(x);
      }
#endif
    };
  } // namespace detail
  /// \brief Function exp for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::exp` applied to the object (nda::Scalar).
  template <Array A>
  auto exp(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::exp_f{})(std::forward<A>(a));
  }

  namespace detail {
    struct cos_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::cos;
        return cos(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires PreservesScalarType<cos_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::cos(x);
      }
#endif
    };
  } // namespace detail
  /// \brief Function cos for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::cos` applied to the object (nda::Scalar).
  template <Array A>
  auto cos(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::cos_f{})(std::forward<A>(a));
  }

  namespace detail {
    struct sin_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::sin;
        return sin(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires PreservesScalarType<sin_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::sin(x);
      }
#endif
    };
  } // namespace detail
  /// \brief Function sin for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::sin` applied to the object (nda::Scalar).
  template <Array A>
  auto sin(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::sin_f{})(std::forward<A>(a));
  }

  namespace detail {
    struct tan_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::tan;
        return tan(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires PreservesScalarType<tan_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::tan(x);
      }
#endif
    };
  } // namespace detail
  /// \brief Function tan for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::tan` applied to the object (nda::Scalar).
  template <Array A>
  auto tan(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::tan_f{})(std::forward<A>(a));
  }

  namespace detail {
    struct cosh_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::cosh;
        return cosh(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires PreservesScalarType<cosh_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::cosh(x);
      }
#endif
    };
  } // namespace detail
  /// \brief Function cosh for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::cosh` applied to the object (nda::Scalar).
  template <Array A>
  auto cosh(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::cosh_f{})(std::forward<A>(a));
  }

  namespace detail {
    struct sinh_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::sinh;
        return sinh(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires PreservesScalarType<sinh_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::sinh(x);
      }
#endif
    };
  } // namespace detail
  /// \brief Function sinh for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::sinh` applied to the object (nda::Scalar).
  template <Array A>
  auto sinh(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::sinh_f{})(std::forward<A>(a));
  }

  namespace detail {
    struct tanh_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::tanh;
        return tanh(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires PreservesScalarType<tanh_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::tanh(x);
      }
#endif
    };
  } // namespace detail
  /// \brief Function tanh for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::tanh` applied to the object (nda::Scalar).
  template <Array A>
  auto tanh(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::tanh_f{})(std::forward<A>(a));
  }

  namespace detail {
    struct acos_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::acos;
        return acos(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires PreservesScalarType<acos_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::acos(x);
      }
#endif
    };
  } // namespace detail
  /// \brief Function acos for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::acos` applied to the object (nda::Scalar).
  template <Array A>
  auto acos(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::acos_f{})(std::forward<A>(a));
  }

  namespace detail {
    struct asin_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::asin;
        return asin(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires PreservesScalarType<asin_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::asin(x);
      }
#endif
    };
  } // namespace detail
  /// \brief Function asin for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::asin` applied to the object (nda::Scalar).
  template <Array A>
  auto asin(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::asin_f{})(std::forward<A>(a));
  }

  namespace detail {
    struct atan_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::atan;
        return atan(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires PreservesScalarType<atan_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::atan(x);
      }
#endif
    };
  } // namespace detail
  /// \brief Function atan for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::atan` applied to the object (nda::Scalar).
  template <Array A>
  auto atan(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::atan_f{})(std::forward<A>(a));
  }

  namespace detail {
    struct log_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::log;
        return log(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires PreservesScalarType<log_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::log(x);
      }
#endif
    };
  } // namespace detail
  /// \brief Function log for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::log` applied to the object (nda::Scalar).
  template <Array A>
  auto log(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::log_f{})(std::forward<A>(a));
  }

  namespace detail {
    struct sqrt_f {
      FORCEINLINE auto operator()(auto const &x) const {
        using std::sqrt;
        return sqrt(x);
      }
#ifdef NDA_HAVE_XSIMD
      template <SimdRealOrComplex T>
        requires PreservesScalarType<sqrt_f, T>
      FORCEINLINE native_simd<T> load(native_simd<T> const &x) const {
        return xsimd::sqrt(x);
      }
#endif
    };
  } // namespace detail
  /// \brief Function sqrt for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::sqrt` applied to the object (nda::Scalar).
  template <Array A>
  auto sqrt(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(detail::sqrt_f{})(std::forward<A>(a));
  }


  /** @} */

} // namespace nda
