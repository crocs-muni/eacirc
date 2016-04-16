#pragma once

#include "base.h"

int count_trailing_zeros(u32 x) {
#ifdef __GNUC__
    return __builtin_ctz(x);
#elif _MSC_VER
    return __lzcnt(x);
#elif __CUDACC__
    return __ffs(*reinterpret_cast<i32*>(&x)) - 1;
#endif
}

int count_trailing_zeros(u64 x) {
#ifdef __GNUC__
    return __builtin_ctzll(x);
#elif _MSC_VER
    return __lzcnt64(x);
#elif __CUDACC__
    return __ffsll(*reinterpret_cast<i64*>(&x)) - 1;
#endif
}


/**
  * Compile-time base 2 logarithm, ie Log2<8>::value is 3.
  */
// clang-format off
template <unsigned I> struct Log2
{ const static unsigned value = 1 + Log2<I / 2>::value; };

template <> struct Log2<1>
{ const static unsigned value = 1; };
// clang-format on

/**
 * Compile-time maximum of the given values, ie. Max<1,9,6>::value is 9.
 */
template <unsigned... I> struct Max;
template <unsigned I> struct Max<I> { const static unsigned value = I; };
template <unsigned I, unsigned... Is> struct Max<I, Is...> {
    const static unsigned value = I > Max<Is...>::value ? I : Max<Is...>::value;
};
