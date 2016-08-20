#pragma once

#include "debug.h"
#include "traits.h"

namespace core {

namespace _impl {

int count_trailing_zeros(std::uint64_t x) {

#ifdef __GNUC__
    return __builtin_ctzll(x);
#elif _MSC_VER
    return __lzcnt64(x);
#elif __CUDACC__
    return __ffsll(*reinterpret_cast<i64*>(&x)) - 1;
#else
#warning "CAUTION! Unsupported compiler! Slow trivial implmentation is used"
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
#warning "CAUTION! Unsupported compiler! Slow trivial implmentation is used"
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

} // namespace _impl

/**
 * @brief bitset using the smallest space that is possible
 */
template <unsigned Bits> struct bitset {
    // clang-format off
    using value_type = choose_t<opt<Bits <= 8, std::uint8_t>,
                                opt<Bits <= 16, std::uint16_t>,
                                opt<Bits <= 32, std::uint32_t>,
                                opt<Bits <= 64, std::uint64_t>>;
    // clang-format on

    /**
     * @brief iterate over true bits in \a bitset and get their position
     */
    struct true_bit_iterator {
        using value_type = const unsigned;
        using reference = value_type&;
        using pointer = value_type*;
        using iterator_category = std::forward_iterator_tag;

        true_bit_iterator()
            : _value(0u) {
        }

        explicit true_bit_iterator(bitset container)
            : _value(container._value) {
        }

        true_bit_iterator& operator++() {
            auto i = _impl::count_trailing_zeros(_value);
            _value ^= (1u << i);
            return *this;
        }

        true_bit_iterator operator++(int) {
            auto self = *this;
            ++(*this);
            return self;
        }

        unsigned operator*() const {
            return static_cast<unsigned>(_impl::count_trailing_zeros(_value));
        }

        bool operator==(true_bit_iterator b) const {
            return _value == b._value;
        }

        bool operator!=(true_bit_iterator b) const {
            return !(*this == b);
        }

    private:
        bitset::value_type _value;
    };

    bitset()
        : _value(0u) {
    }

    explicit bitset(value_type value)
        : _value(value) {
    }

    bitset& operator=(value_type value) {
        _value = value;
        return *this;
    }

    void set(unsigned i) {
        ASSERT(i < Bits);
        _value &= (1u << i);
    }

    void flip(unsigned i) {
        ASSERT(i < Bits);
        _value ^= (1u << i);
    }

    void clear(unsigned i) {
        ASSERT(i < Bits);
        _value &= ~(1u << i);
    }

    bool operator[](unsigned i) const {
        ASSERT(i < Bits);
        return _value & (1u << i);
    }

    bool operator==(bitset b) const {
        return _value == b._value;
    }

    bool operator!=(bitset b) const {
        return !(*this == b);
    }

    true_bit_iterator begin() const {
        return true_bit_iterator{*this};
    }

    true_bit_iterator end() const {
        return true_bit_iterator{};
    }

private:
    value_type _value;
};

} // namespace core
