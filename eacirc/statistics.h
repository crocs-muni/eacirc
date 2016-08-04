#pragma once

#include "dataset.h"
#include <cmath>
#include <vector>

struct two_sample_chisqr {
    two_sample_chisqr(std::size_t categories)
        : _histogram_a(categories)
        , _histogram_b(categories) {}

    template <std::size_t Size>
    double operator()(const dataset<Size> &data_a,
                      const dataset<Size> &data_b) {
        std::fill(_histogram_a.begin(), _histogram_a.end(), 0u);
        std::fill(_histogram_b.begin(), _histogram_b.end(), 0u);

        for (auto &vec : data_a)
            for (std::uint8_t byte : vec)
                _histogram_a[byte % _histogram_a.size()]++;
        for (auto &vec : data_b)
            for (std::uint8_t byte : vec)
                _histogram_b[byte % _histogram_b.size()]++;
        return _compute();
    }

private:
    std::vector<std::uint64_t> _histogram_a;
    std::vector<std::uint64_t> _histogram_b;

    double _compute() const;
};
