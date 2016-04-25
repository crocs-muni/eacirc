#pragma once

#include "base.h"

int count_trailing_zeros(u64 x) {
#ifdef __GNUC__
    return __builtin_ctzll(x);
#elif _MSC_VER
    return __lzcnt64(x);
#elif __CUDACC__
    return __ffsll(*reinterpret_cast<i64*>(&x)) - 1;
#endif
}

int count_trailing_zeros(u32 x) {
#ifdef __GNUC__
    return __builtin_ctz(x);
#elif _MSC_VER
    return __lzcnt(x);
#elif __CUDACC__
    return __ffs(*reinterpret_cast<i32*>(&x)) - 1;
#endif
}

int count_trailing_zeros(u16 x) {
    return count_trailing_zeros(static_cast<u32>(x));
}

int count_trailing_zeros(u8 x) {
    return count_trailing_zeros(static_cast<u32>(x));
}
