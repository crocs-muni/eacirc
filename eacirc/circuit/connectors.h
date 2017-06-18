#pragma once

#include <eacirc-core/debug.h>
#include <cstdint>

namespace circuit {

    namespace _impl {

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


        int count_true_bits(std::uint8_t x) {
#ifdef __GNUC__
            return __builtin_popcount(x);
#elif _MSC_VER
            return __popcnt16(x);
#else
#warning "CAUTION! Unsupported compiler! Slow trivial implementation is used"
            int count = 0;
            while (x) {
                if (x % 2)
                    ++count;
                x >>= 1;
            }
            return count;
#endif
        }

        int count_trailing_zeros(std::uint16_t x) {
            return count_trailing_zeros(static_cast<std::uint32_t>(x));
        }

        int count_trailing_zeros(std::uint8_t x) {
            return count_trailing_zeros(static_cast<std::uint32_t>(x));
        }

    } // namespace _impl

    template <unsigned> struct store;
    template <> struct store<8> { using type = std::uint8_t; };
    template <> struct store<16> { using type = std::uint16_t; };
    template <> struct store<32> { using type = std::uint32_t; };
    template <> struct store<64> { using type = std::uint64_t; };

    template <typename T> struct connector_iterator {
        connector_iterator(T mask)
            : _mask(mask) {}

        void next() {
            auto i = _impl::count_trailing_zeros(_mask);
            _mask ^= (1u << i);
        }

        bool has_next() const { return _mask != 0u; }

        operator unsigned() const {
            return static_cast<unsigned>(_impl::count_trailing_zeros(_mask));
        }

    private:
        T _mask;
    };

    template <unsigned Size> struct connectors {
        static constexpr unsigned size = Size;

        using value_type = typename store<size>::type;

        connectors(value_type mask)
            : _mask(mask) {}

        void set(unsigned i) {
            ASSERT(i < size);
            _mask |= (1u << i);
        }

        void flip(unsigned i) {
            ASSERT(i < size);
            _mask ^= (1u << i);
        }

        void clear(unsigned i) {
            ASSERT(i < size);
            _mask &= ~(1u << i);
        }

        std::size_t count_connectors() {
            return std::size_t(_impl::count_true_bits(_mask));
        }

        connector_iterator<value_type> iterator() const { return _mask; }

        bool operator[](unsigned i) const {
            ASSERT(i < size);
            return _mask & (1u << i);
        }

        bool operator==(connectors rhs) const { return _mask == rhs._mask; }
        bool operator!=(connectors rhs) const { return _mask != rhs._mask; }

    private:
        value_type _mask;
    };

} // namespace circuit
