#pragma once

#include <algorithm>
#include <array>
#include <random>

template <class T, std::size_t S> struct sample_pool {
    sample_pool(std::initializer_list<T> samples)
        : _size(samples.size()) {
        std::copy(samples.begin(), samples.end(), _pool.begin());
    }

    template <class Generator> const T &operator()(Generator &g) const {
        std::uniform_int_distribution<std::size_t> dis{0, _size};
        return _pool[dis(g)];
    }

private:
    std::size_t _size;
    std::array<T, S> _pool;
};
