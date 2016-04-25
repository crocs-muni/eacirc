#pragma once

#include "base.h"
#include "utils.h"
#include <iterator>
#include <type_traits>

// clang-format off
namespace detail {
template <unsigned Bits>
struct Store : std::conditional_t<(Bits > 8),
                       std::conditional_t<(Bits > 16),
                               std::conditional_t<(Bits > 32), Store<64>, Store<32>>,
                               Store<16>>,
                       Store<8>> {
    static_assert(Bits > 64, "Cannot create storage bigger than 64 bits, use std::bitset instead.");
};

template <> struct Store<8> { using Type = u8; };
template <> struct Store<16> { using Type = u16; };
template <> struct Store<32> { using Type = u32; };
template <> struct Store<64> { using Type = u64; };
} // namespace detail
// clang-format on

/**
 * Unsigned integer storage parametrized by it's size in bits.
 *
 * @param Bits the size of the word in bits
 */
template <unsigned Bits> using Store = typename detail::Store<Bits>::Type;

/**
 * Iterator which iterates over true bits in bit mask from lower to higher
 * position.
 *
 * @param Bits the size of the bit mask in bits
 */
template <unsigned Bits>
struct BitIterator : std::iterator<std::forward_iterator_tag, const unsigned> {

public:
    BitIterator() = default;
    BitIterator(Store<Bits> mask) : _mask(mask) {}

    BitIterator& operator++() {
        auto i = ::count_trailing_zeros(_mask);
        _mask ^= 1u << i;
        return *this;
    }

    BitIterator operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

    unsigned operator*() const { return static_cast<unsigned>(::count_trailing_zeros(_mask)); }

    bool operator==(BitIterator const& b) const { return _mask == b._mask; }
    bool operator!=(BitIterator const& b) const { return !(*this == b); }

private:
    Store<Bits> _mask{0u};
};

/**
 * Bit mask parametrized by it's size
 *
 * @param Bits the size of the storage in bits
 */
template <unsigned Bits> struct Bitset {
    using Type = Store<Bits>;

private:
    Type _mask{0u};

public:
    Bitset() = default;
    Bitset(Type mask) : _mask(mask) {}

    bool operator[](unsigned i) const { return _mask & (1u << i); }

    void set(unsigned i) { _mask &= (1u << i); }
    void flip(unsigned i) { _mask ^= 1u << i; }
    void clear(unsigned i) { _mask &= ~(1u << i); }

    bool operator==(Bitset b) const { return _mask == b._mask; }
    bool operator!=(Bitset b) const { return _mask != b._mask; }

    BitIterator<Bits> begin() const { return _mask; }
    BitIterator<Bits> end() const { return 0u; }
};
