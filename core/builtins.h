#pragma once

#include <cstdint>

namespace core {
namespace builtins {

int count_trailing_zeros(std::uint64_t x) {
#ifdef __GNUC__
  return __builtin_ctzll(x);
#elif _MSC_VER
  return __lzcnt64(x);
#elif __CUDACC__
  return __ffsll(*reinterpret_cast<i64*>(&x)) - 1;
#else
#warning "CAUTION! Unsupported compiler! Slow trivial implementation is used"
  if (x == 0)
    return 64;
  int n = 1;
  if ((x & 0xffffffff) == 0) {
    x >>= 32;
    n += 32;
  }
  if ((x & 0xffff) == 0) {
    x >>= 16;
    n += 16;
  }
  if ((x & 0xff) == 0) {
    x >>= 8;
    n += 8;
  }
  if ((x & 0xf) == 0) {
    x >>= 4;
    n += 4;
  }
  if ((x & 0x3) == 0) {
    x >>= 2;
    n += 2;
  }
  return n -= x & 0x1;
#endif
}

int count_trailing_zeros(std::uint32_t x) {
#ifdef __GNUC__
  return __builtin_ctz(x);
#elif _MSC_VER
  return __lzcnt(x);
#elif __CUDACC__
  return __ffs(*reinterpret_cast<i32*>(&x)) - 1;
#else
#warning "CAUTION! Unsupported compiler! Slow trivial implementation is used"
  if (x == 0)
    return 32;
  int n = 1;
  if ((x & 0xffff) == 0) {
    x >>= 16;
    n += 16;
  }
  if ((x & 0xff) == 0) {
    x >>= 8;
    n += 8;
  }
  if ((x & 0xf) == 0) {
    x >>= 4;
    n += 4;
  }
  if ((x & 0x3) == 0) {
    x >>= 2;
    n += 2;
  }
  return n -= x & 0x1;
#endif
}

int count_trailing_zeros(std::uint16_t x) {
  return count_trailing_zeros(static_cast<std::uint32_t>(x));
}

int count_trailing_zeros(std::uint8_t x) {
  return count_trailing_zeros(static_cast<std::uint32_t>(x));
}

} // namespace builtins
} // namespace core
