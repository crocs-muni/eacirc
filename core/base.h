#pragma once

#include <cinttypes>
#include <cstddef>

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

#include <memory>

namespace std {
template <class T, class... Args> std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace std

#include <type_traits>

namespace std {
template <bool B, class T, class F> using conditional_t = typename conditional<B, T, F>::type;
template <class T> using decay_t = typename decay<T>::type;
template <bool B, class T = void> using enable_if_t = typename enable_if<B, T>::type;
template <class T> using underlying_type_t = typename underlying_type<T>::type;
template <class T> using result_of_t = typename result_of<T>::type;
} // namespace std

/**
 * Various type classes
 */
struct Eq {};
struct Swappable {};

template <class T>
auto operator!=(const T& a, const T& b) -> std::enable_if_t<std::is_base_of<Eq, T>::value, bool> {
    return !(a == b);
}

template <class T> auto swap(T& a, T& b) -> std::enable_if_t<std::is_base_of<Swappable, T>::value> {
    a.swap(b);
}
