#pragma once

#include <algorithm>
#include <core/json.h>
#include <core/stream.h>
#include <limits>
#include <pcg/pcg_random.hpp>
#include <random>

namespace streams {

    /**
     * \brief Stream of true bits
     */
    struct true_stream : stream {
        void read(dataset& data) override {
            std::fill_n(data.data(), data.size(), std::numeric_limits<std::uint8_t>::max());
        }
    };

    /**
     * \brief Stream of false bits (everything is zeroed out)
     */
    struct false_stream : stream {
        void read(dataset& data) override {
            std::fill_n(data.data(), data.size(), std::numeric_limits<std::uint8_t>::min());
        }
    };

    /**
     * \brief Merseine Twister generator
     */
    struct mt19937_stream : stream {
        template <typename Sseq>
        mt19937_stream(Sseq&& seeder)
            : _prng(seeder) {}

        void read(dataset& data) override {
            using uint_type = decltype(_prng)::result_type;

            auto p = reinterpret_cast<uint_type*>(data.data());
            auto n = data.size() / sizeof(uint_type);

            std::generate_n(p, n, _prng);
        }

    private:
        std::mt19937 _prng;
    };

    /**
     * \brief Permutation Congruential generator
     */
    struct pcg32_stream : stream {
        template <typename Sseq>
        pcg32_stream(Sseq&& seeder)
            : _prng(seeder) {}

        void read(dataset& data) override {
            using uint_type = decltype(_prng)::result_type;

            auto p = reinterpret_cast<uint_type*>(data.data());
            auto n = data.size() / sizeof(uint_type);

            std::generate_n(p, n, _prng);
        }

    private:
        pcg32 _prng;
    };

} // namespace streams
