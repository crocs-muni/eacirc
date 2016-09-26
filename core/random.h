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
