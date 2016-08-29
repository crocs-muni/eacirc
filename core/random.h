#pragma once

#include <cstdint>
#include <pcg/pcg_random.hpp>
#include <utility>

template <typename Generator> struct seed_seq_from {
    using result_type = std::uint_least32_t;

    template <typename... Args>
    seed_seq_from(Args&&... args)
        : _rng(std::forward<Args>(args)...) {}

    seed_seq_from(seed_seq_from const&) = delete;
    seed_seq_from& operator=(seed_seq_from const&) = delete;

    template <typename I> void generate(I beg, I end) {
        for (; beg != end; ++beg)
            *beg = result_type(_rng());
    }

    constexpr std::size_t size() const {
        return (sizeof(typename Generator::result_type) > sizeof(result_type) &&
                Generator::max() > ~std::size_t(0UL))
                       ? ~std::size_t(0UL)
                       : size_t(Generator::max());
    }

private:
    Generator _rng;
};

using default_random_generator = pcg32;
using default_seed_source = seed_seq_from<default_random_generator>;

/*
#include <fstream>
#include <limits.h>
#include <pcg/pcg_extras.hpp>
#include <pcg/pcg_random.hpp>
#include <random>

namespace core {

    using default_generator = pcg32;
    using seed_source = pcg_extras::seed_seq_from<pcg32>;

    template <class Type> struct qrng_engine {
        using result_type = Type;

        qrng_engine(const std::string file)
            : _file(file) {
            if (!_file.is_open())
                throw std::runtime_error("Can't open file \"" + file + "\".");
        }

        result_type operator()() {
            result_type value;
            _file.read(reinterpret_cast<std::ifstream::char_type*>(&value));
            if (_file.eof())
                throw std::range_error("reading of qrng data reached end of file.");
            if (_file.fail())
                throw std::runtime_error("an unrecoverable read error during reading of qrng data");
            return value;
        }

        static constexpr result_type min() { return std::numeric_limits<result_type>::min(); }

        static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    private:
        std::ifstream _file;
    };

} // namespace core
*/
