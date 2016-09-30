#pragma once

#include <algorithm>
#include <core/stream.h>
#include <limits>
#include <pcg/pcg_random.hpp>
#include <random>

namespace streams {

    namespace _impl {

        template <byte Value> struct const_stream : stream {
            void read(byte_view out) override { std::fill(out.begin(), out.end(), Value); }
        };

        template <typename Generator> struct prng_stream : stream {
            template <typename Sseq>
            prng_stream(Sseq&& seeder)
                : _prng(std::forward<Sseq>(seeder)) {}

            void read(byte_view out) override {
                using type = typename Generator::result_type;

                auto p = reinterpret_cast<type*>(out.begin());
                auto n = out.size() / sizeof(type);

                std::generate_n(
                        p, n, [this] { return std::uniform_int_distribution<type>()(_prng); });
            }

        private:
            Generator _prng;
        };

    } // namespace _impl

    /**
     * \brief Stream of true bits
     */
    using true_stream = _impl::const_stream<std::numeric_limits<byte>::max()>;

    /**
     * \brief Stream of false bits
     */
    using false_stream = _impl::const_stream<std::numeric_limits<byte>::min()>;

    /**
     * \brief Stream of data produced by Merseine Twister
     */
    using mt19937_stream = _impl::prng_stream<std::mt19937>;

    /**
     * \brief Stream of data produced by PCG (Permutation Congruential Generator)
     */
    using pcg32_stream = _impl::prng_stream<pcg32>;

} // namespace streams
