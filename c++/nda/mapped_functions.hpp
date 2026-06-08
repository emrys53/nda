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

    // Functor for nda::detail::conj.
    struct conj_f {
      auto operator()(auto const &x) const { return conj(x); };

      auto load(auto const &x) const {
        //TODO: We need to change this check after custom complex class.
        if constexpr (xsimd::is_batch_complex<std::remove_cvref_t<decltype(x)>>::value) {
          return xsimd::conj(x);
        } else {
          return x;
        }
      }
    };

    struct pow_f {
      double exponent;
      FORCEINLINE auto operator()(auto const &x) const {
        using std::pow;
        return pow(x, exponent);
      }
      FORCEINLINE auto load(auto const &x) const {
        using xsimd::pow;
        using simd_t = std::remove_cvref_t<decltype(x)>;
        return pow(x, simd_t(exponent));
      }
    };

    template <typename F>
    struct unary_functor {
      FORCEINLINE auto operator()(auto const &x) const { return F{}(x); }
      FORCEINLINE auto load(auto const &x) const { return F{}(x); }
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
    return nda::map([](auto const &x) {
      if constexpr (Scalar<decltype(x)>) {
        return 1.0 / x;
      } else {
        return reciprocal(x);
      }
    })(std::forward<A>(a));
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
    requires(((Scalar<A> && Scalar<B>) || (Array<A> && Array<B> && get_rank<A> == get_rank<B>))
             && !is_complex_v<get_value_t<A>> && !is_complex_v<get_value_t<B>>)
  [[nodiscard]] auto max(A &&a, B &&b) {
    return nda::map([](auto const &x, auto const &y) {
      using std::max;
      return max(x, y);
    })(std::forward<A>(a), std::forward<B>(b));
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
    requires(((Scalar<A> && Scalar<B>) || (Array<A> && Array<B> && get_rank<A> == get_rank<B>))
             && !is_complex_v<get_value_t<A>> && !is_complex_v<get_value_t<B>>)
  [[nodiscard]] auto min(A &&a, B &&b) {
    return nda::map([](auto const &x, auto const &y) {
      using std::min;
      return min(x, y);
    })(std::forward<A>(a), std::forward<B>(b));
  }



  /// \brief Function abs for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::abs` applied to the object (nda::Scalar).
  template <Array A>
  auto abs(A &&a)  {
    return nda::map([](auto const &x) {
      using std::abs;
      return abs(x);
    })(std::forward<A>(a));
  }

  /// \brief Function imag for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::imag` applied to the object (nda::Scalar).
  template <ArrayOrScalar A>
  auto imag(A &&a)  {
    return nda::map(
       [](auto const &x) {
         using std::imag;
         return imag(x);
       })(std::forward<A>(a));
  }

  /// \brief Function floor for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::floor` applied to the object (nda::Scalar).
  template <Array A>
  auto floor(A &&a)  {
    auto lambda = [](auto const &x) {
      using xsimd::floor;
      return floor(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }

  /// \brief Function real for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type..
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `nda::detail::real` applied to the object (nda::Scalar).
  template <ArrayOrScalar A>
  auto real(A &&a) {
    return nda::map(
       [](auto const &x) {return detail::real(x); })(std::forward<A>(a));
  }

  /// \brief Function abs2 for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type..
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `nda::detail::abs2` applied to the object (nda::Scalar).
  template <ArrayOrScalar A>
  auto abs2(A &&a) {
    return nda::map(
       [](auto const &x) {return detail::abs2(x); })(std::forward<A>(a));
  }

  /// \brief Function isnan for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type..
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `nda::detail::isnan` applied to the object (nda::Scalar).
  template <typename A>
    requires(Array<A> or is_complex_v<A>)
  auto isnan(A &&a) {
    return nda::map(
       [](auto const &x) {return detail::isnan(x); })(std::forward<A>(a));
  }

  /// \brief Function exp for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return  A lazy nda::expr_call object (nda::Array) or the result of `std::exp` applied to the object (nda::Scalar).
  template <Array A>
  auto exp(A &&a) requires(get_algebra<A> != 'M') {
    auto lambda = [](auto const &x) {
      using xsimd::exp;
      return exp(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }

  /// \brief Function cos for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return  A lazy nda::expr_call object (nda::Array) or the result of `std::cos` applied to the object (nda::Scalar).
  template <Array A>
  auto cos(A &&a) requires(get_algebra<A> != 'M') {
    auto lambda = [](auto const &x) {
      using xsimd::cos;
      return cos(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }

  /// \brief Function sin for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return  A lazy nda::expr_call object (nda::Array) or the result of `std::sin` applied to the object (nda::Scalar).
  template <Array A>
  auto sin(A &&a) requires(get_algebra<A> != 'M') {
    auto lambda = [](auto const &x) {
      using xsimd::sin;
      return sin(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }

  /// \brief Function tan for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return  A lazy nda::expr_call object (nda::Array) or the result of `std::tan` applied to the object (nda::Scalar).
  template <Array A>
  auto tan(A &&a) requires(get_algebra<A> != 'M') {
    auto lambda = [](auto const &x) {
      using xsimd::tan;
      return tan(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }

  /// \brief Function cosh for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return  A lazy nda::expr_call object (nda::Array) or the result of `std::cosh` applied to the object (nda::Scalar).
  template <Array A>
  auto cosh(A &&a) requires(get_algebra<A> != 'M') {
    auto lambda = [](auto const &x) {
      using xsimd::cosh;
      return cosh(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }

  /// \brief Function sinh for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return  A lazy nda::expr_call object (nda::Array) or the result of `std::sinh` applied to the object (nda::Scalar).
  template <Array A>
  auto sinh(A &&a) requires(get_algebra<A> != 'M') {
    auto lambda = [](auto const &x) {
      using xsimd::sinh;
      return sinh(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }

  /// \brief Function tanh for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return  A lazy nda::expr_call object (nda::Array) or the result of `std::tanh` applied to the object (nda::Scalar).
  template <Array A>
  auto tanh(A &&a) requires(get_algebra<A> != 'M') {
    auto lambda = [](auto const &x) {
      using xsimd::tanh;
      return tanh(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }

  /// \brief Function acos for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return  A lazy nda::expr_call object (nda::Array) or the result of `std::acos` applied to the object (nda::Scalar).
  template <Array A>
  auto acos(A &&a) requires(get_algebra<A> != 'M') {
    auto lambda = [](auto const &x) {
      using xsimd::acos;
      return acos(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }

  /// \brief Function asin for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return  A lazy nda::expr_call object (nda::Array) or the result of `std::asin` applied to the object (nda::Scalar).
  template <Array A>
  auto asin(A &&a) requires(get_algebra<A> != 'M') {
    auto lambda = [](auto const &x) {
      using xsimd::asin;
      return asin(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }

  /// \brief Function atan for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return  A lazy nda::expr_call object (nda::Array) or the result of `std::atan` applied to the object (nda::Scalar).
  template <Array A>
  auto atan(A &&a) requires(get_algebra<A> != 'M') {
    auto lambda = [](auto const &x) {
      using xsimd::atan;
      return atan(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }

  /// \brief Function log for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return  A lazy nda::expr_call object (nda::Array) or the result of `std::log` applied to the object (nda::Scalar).
  template <Array A>
  auto log(A &&a) requires(get_algebra<A> != 'M') {
    auto lambda = [](auto const &x) {
      using xsimd::log;
      return log(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }

  /// \brief Function sqrt for non-matrix nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return  A lazy nda::expr_call object (nda::Array) or the result of `std::sqrt` applied to the object (nda::Scalar).
  template <Array A>
  auto sqrt(A &&a) requires(get_algebra<A> != 'M') {
    auto lambda = [](auto const &x) {
      using xsimd::sqrt;
      return sqrt(x);
    };

    return nda::map(detail::unary_functor<decltype(lambda)>{})(std::forward<A>(a));
  }


  /** @} */

} // namespace nda
