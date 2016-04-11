#pragma once

#include <cinttypes>
#include <type_traits>

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using ui8 = std::uint8_t;
using ui16 = std::uint16_t;
using ui32 = std::uint32_t;
using ui64 = std::uint64_t;

namespace detail {
    template <unsigned> struct Store;

    // clang-format off
    template <> struct Store<8>  { using Type = ui8;  };
    template <> struct Store<16> { using Type = ui16; };
    template <> struct Store<32> { using Type = ui32; };
    template <> struct Store<64> { using Type = ui64; };
    //clang-format on
}

/**
 * Unsigned integer storage parametrized by it's size in bits
 * @param Bits the size of the word in bits
 */
template <unsigned Bits> using Store = typename detail::Store<Bits>::Type;


/**
 * Simple typedef semanticaly denoting ownership of a resource (memory, atc..).
 * A siple use is 'Owner<int*> memory = new int', which express that some memory must be freed later in the code.
 * This idiom is rocomended by Bjarne Stroustrup (creator of C++), it produces more readable code which can be statically checked for memory leaks.
 */
template <class T>
using Owner = T;

/**
 * Performs static_cast of an scoped enum to it's underlying type
 */
template <class E>
constexpr auto to_underlying(E e) -> typename std::underlying_type<E>::type
{
    return static_cast<typename std::underlying_type<E>::type>(e);
}
