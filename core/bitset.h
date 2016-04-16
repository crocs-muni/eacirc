#pragma once

#include "base.h"
#include "utils.h"
#include <iterator>

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

    unsigned operator*() const {
        return static_cast<unsigned>(::count_trailing_zeros(_mask));
    }

    bool operator==(BitIterator& b) const { return _mask == b._mask; }
    bool operator!=(BitIterator& b) const { return !(*this == b); }

private:
    Store<Bits> _mask{0u};
};

/**
 * Bit mask parametrized by it's size
 *
 * @param Bits the size of the storage in bits
 */
template <unsigned Bits> class Bitset {
    Store<Bits> _mask{0u};

public:
    Bitset() = default;
    Bitset(Store<Bits> mask) : _mask(mask) {}

    bool operator[](unsigned i) const { return _mask & (1u << i); }

    void set(unsigned i) { _mask &= (1u << i); }
    void toggle(unsigned i) { _mask ^= 1u << i; }
    void clear(unsigned i) { _mask &= ~(1u << i); }

    bool operator==(Bitset b) const { return _mask == b._mask; }
    bool operator!=(Bitset b) const { return _mask != b._mask; }

    BitIterator<Bits> begin() const { return _mask; }
    BitIterator<Bits> end() const { return 0u; }
};
