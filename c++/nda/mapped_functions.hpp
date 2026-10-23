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
#include "./macros.hpp"
#include "./map.hpp"
#include "./simd/simd_concepts.hpp"
#include "./traits.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdlib>
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

    // Get the complex conjugate of a scalar.
    template <nda::Scalar S>
    auto conj(S x) {
      if constexpr (is_complex_v<S>) {
        return std::conj(x);
      } else {
        return x;
      }
    }

    // Get the squared absolute value with the return type of std::norm.
    FORCEINLINE auto abs2(auto x) { return std::norm(x); }

    // Check if a scalar or either component of a complex scalar is NaN.
    template <Scalar S>
    bool isnan(S const &x) {
      if constexpr (is_complex_v<S>) {
        return std::isnan(x.real()) or std::isnan(x.imag());
      } else {
        return std::isnan(x);
      }
    }

    // Functors of the mapped functions below. operator() is the scalar call; load() is the batch version, constrained
    // to the scalar domain xsimd supports (see the Simd* concepts). load() is a template on the native batch type
    // (nda::simd::NativeSimd, never satisfied without xsimd) and calls xsimd's functions unqualified, so that they are
    // found by argument-dependent lookup and the functors compile without xsimd. nda::map applies operator() element by
    // element and, if the functor has a load for the argument batches, load() in SIMD blocks (see
    // nda::has_simd_load_v); a functor without a matching load keeps the expression on the scalar path.
    struct conj_f {
      [[nodiscard]] FORCEINLINE auto operator()(auto const &x) const noexcept { return conj(x); }

      template <simd::NativeSimd B>
        requires(simd::SimdArithmetic<typename B::value_type> or simd::SimdFloatingPointOrComplex<typename B::value_type>)
      [[nodiscard]] FORCEINLINE auto load(B const &x) const noexcept {
        if constexpr (is_complex_v<typename B::value_type>) {
          return conj(x);
        } else {
          return x;
        }
      }
    };

    struct real_f {
      [[nodiscard]] FORCEINLINE auto operator()(auto const &x) const noexcept { return detail::real(x); }

      template <simd::NativeSimd B>
        requires((simd::SimdArithmetic<typename B::value_type> or simd::SimdFloatingPointOrComplex<typename B::value_type>)
                 and simd::SimdPreservesLaneWidth<remove_complex_t<typename B::value_type>, typename B::value_type>)
      [[nodiscard]] FORCEINLINE auto load(B const &x) const noexcept {
        if constexpr (is_complex_v<typename B::value_type>) {
          return real(x);
        } else {
          return x;
        }
      }
    };

    struct imag_f {
      [[nodiscard]] FORCEINLINE auto operator()(auto const &x) const noexcept {
        using std::imag;
        return imag(x);
      }

      template <simd::NativeSimd B>
        requires(simd::SimdFloatingPointOrComplex<typename B::value_type>
                 and simd::SimdPreservesLaneWidth<remove_complex_t<typename B::value_type>, typename B::value_type>)
      [[nodiscard]] FORCEINLINE auto load(B const &x) const noexcept {
        if constexpr (is_complex_v<typename B::value_type>) {
          return imag(x);
        } else {
          return B(0);
        }
      }
    };

    struct abs2_f {
      [[nodiscard]] FORCEINLINE auto operator()(auto const &x) const noexcept { return detail::abs2(x); }

      template <simd::NativeSimd B>
        requires(simd::SimdFloatingPointOrComplex<typename B::value_type>
                 and simd::SimdPreservesLaneWidth<remove_complex_t<typename B::value_type>, typename B::value_type>)
      [[nodiscard]] FORCEINLINE auto load(B const &x) const noexcept {
        if constexpr (is_complex_v<typename B::value_type>) {
          return norm(x);
        } else {
          return x * x;
        }
      }
    };

    struct abs_f {
      [[nodiscard]] FORCEINLINE auto operator()(auto const &x) const noexcept {
        if constexpr (std::is_unsigned_v<std::remove_cvref_t<decltype(x)>>) {
          return x;
        } else {
          using std::abs;
          return abs(x);
        }
      }

      template <simd::NativeSimd B>
        requires((simd::SimdSignedArithmetic<typename B::value_type> or simd::SimdComplex<typename B::value_type>)
                 and simd::SimdPreservesLaneWidth<remove_complex_t<typename B::value_type>, typename B::value_type>)
      [[nodiscard]] FORCEINLINE auto load(B const &x) const noexcept {
        return abs(x);
      }
    };

    struct floor_f {
      [[nodiscard]] FORCEINLINE auto operator()(auto const &x) const noexcept {
        using std::floor;
        return floor(x);
      }

      template <simd::NativeSimd B>
        requires simd::SimdFloatingPoint<typename B::value_type>
      [[nodiscard]] FORCEINLINE auto load(B const &x) const noexcept {
        return floor(x);
      }
    };

    template <typename S = double>
    struct reciprocal_f {
      S alpha{1};
      [[nodiscard]] FORCEINLINE auto operator()(auto const &x) const noexcept {
        if constexpr (Scalar<std::remove_cvref_t<decltype(x)>>) {
          return alpha / x;
        } else {
          return alpha * reciprocal(x); // nested arrays: nda::reciprocal, found by ADL at instantiation
        }
      }

      template <simd::NativeSimd B>
        requires simd::SimdFloatingPointOrComplex<typename B::value_type>
      [[nodiscard]] FORCEINLINE auto load(B const &x) const noexcept {
        return B(static_cast<typename B::value_type>(alpha)) / x;
      }
    };

    template <typename P = double>
    struct pow_f {
      P exponent;
      [[nodiscard]] FORCEINLINE auto operator()(auto const &x) const noexcept {
        using std::pow;
        return pow(x, exponent);
      }

      // Batches only where std::pow(T, P), T the value type of the batch, has type T: an exponent of the real precision
      // of T or of type T, or any arithmetic exponent for double T. Integer exponents use exponentiation by squaring; a complex base with an
      // integer exponent is a distinct std::pow overload and stays on the scalar path.
      template <simd::NativeSimd B>
        requires(simd::SimdFloatingPointOrComplex<typename B::value_type>
                 and (std::same_as<P, remove_complex_t<typename B::value_type>> or std::same_as<P, typename B::value_type>
                      or (std::same_as<typename B::value_type, double> and std::is_arithmetic_v<P>)))
      [[nodiscard]] FORCEINLINE auto load(B const &x) const noexcept {
        if constexpr (std::is_integral_v<P>) {
          return pow(x, exponent); // optimized integer exponentation.
        } else {
          return pow(x, B(exponent));
        }
      }
    };

    struct max_f {
      [[nodiscard]] FORCEINLINE auto operator()(auto const &x, auto const &y) const noexcept {
        using std::max;
        return max(x, y);
      }

      template <simd::NativeSimd B>
        requires simd::SimdArithmetic<typename B::value_type>
      [[nodiscard]] FORCEINLINE auto load(B const &x, B const &y) const noexcept {
        return max(x, y);
      }
    };

    struct min_f {
      [[nodiscard]] FORCEINLINE auto operator()(auto const &x, auto const &y) const noexcept {
        using std::min;
        return min(x, y);
      }

      template <simd::NativeSimd B>
        requires simd::SimdArithmetic<typename B::value_type>
      [[nodiscard]] FORCEINLINE auto load(B const &x, B const &y) const noexcept {
        return min(x, y);
      }
    };

    struct mul_f {
      [[nodiscard]] FORCEINLINE auto operator()(auto const &x, auto const &y) const noexcept(noexcept(x * y)) { return x * y; }

      template <simd::NativeSimd B>
        requires simd::Vectorizable<typename B::value_type>
      [[nodiscard]] FORCEINLINE auto load(B const &x, B const &y) const noexcept {
        return x * y;
      }
    };

// Functor of a mapped function with the same name in std and in xsimd (batches of float, double and their complex types).
#define NDA_DETAIL_MATH_FUNCTOR(F)                                                                                                                   \
  struct F##_f {                                                                                                                                     \
    [[nodiscard]] FORCEINLINE auto operator()(auto const &x) const noexcept {                                                                        \
      using std::F;                                                                                                                                  \
      return F(x);                                                                                                                                   \
    }                                                                                                                                                \
                                                                                                                                                     \
    template <simd::NativeSimd B>                                                                                                                    \
      requires simd::SimdFloatingPointOrComplex<typename B::value_type>                                                                              \
    [[nodiscard]] FORCEINLINE auto load(B const &x) const noexcept {                                                                                 \
      return F(x);                                                                                                                                   \
    }                                                                                                                                                \
  };

    NDA_DETAIL_MATH_FUNCTOR(exp)
    NDA_DETAIL_MATH_FUNCTOR(cos)
    NDA_DETAIL_MATH_FUNCTOR(sin)
    NDA_DETAIL_MATH_FUNCTOR(tan)
    NDA_DETAIL_MATH_FUNCTOR(cosh)
    NDA_DETAIL_MATH_FUNCTOR(sinh)
    NDA_DETAIL_MATH_FUNCTOR(tanh)
    NDA_DETAIL_MATH_FUNCTOR(acos)
    NDA_DETAIL_MATH_FUNCTOR(asin)
    NDA_DETAIL_MATH_FUNCTOR(atan)
    NDA_DETAIL_MATH_FUNCTOR(log)
    NDA_DETAIL_MATH_FUNCTOR(sqrt)

#undef NDA_DETAIL_MATH_FUNCTOR

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
    if constexpr (is_complex_v<get_value_t<A>>)
      return nda::map(detail::conj_f{})(std::forward<A>(a));
    else
      return std::forward<A>(a);
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
      using real_t      = remove_complex_t<get_value_t<A>>;
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
    requires(((Scalar<A> && Scalar<B>) || (Array<A> && Array<B> && get_rank<A> == get_rank<B>))
             && !is_complex_v<get_value_t<A>> && !is_complex_v<get_value_t<B>>)
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
    requires(((Scalar<A> && Scalar<B>) || (Array<A> && Array<B> && get_rank<A> == get_rank<B>))
             && !is_complex_v<get_value_t<A>> && !is_complex_v<get_value_t<B>>)
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
  /// \details Elements of an unsigned type are returned unchanged.
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
    requires(Array<A> || is_complex_v<A>)
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
  auto exp(A &&a)
    requires(get_algebra<A> != 'M')
  {
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
  auto cos(A &&a)
    requires(get_algebra<A> != 'M')
  {
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
  auto sin(A &&a)
    requires(get_algebra<A> != 'M')
  {
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
  auto tan(A &&a)
    requires(get_algebra<A> != 'M')
  {
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
  auto cosh(A &&a)
    requires(get_algebra<A> != 'M')
  {
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
  auto sinh(A &&a)
    requires(get_algebra<A> != 'M')
  {
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
  auto tanh(A &&a)
    requires(get_algebra<A> != 'M')
  {
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
  auto acos(A &&a)
    requires(get_algebra<A> != 'M')
  {
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
  auto asin(A &&a)
    requires(get_algebra<A> != 'M')
  {
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
  auto atan(A &&a)
    requires(get_algebra<A> != 'M')
  {
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
  auto log(A &&a)
    requires(get_algebra<A> != 'M')
  {
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
  auto sqrt(A &&a)
    requires(get_algebra<A> != 'M')
  {
    return nda::map(detail::sqrt_f{})(std::forward<A>(a));
  }

  /** @} */

} // namespace nda
