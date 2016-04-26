#pragma once

#include <core/base.h>
#include <vector>

class Categories {
    const unsigned _precision;
    std::vector<u64> _histogram_A;
    std::vector<u64> _histogram_B;

public:
    Categories(unsigned precision)
        : _precision(precision), _histogram_A(_precision), _histogram_B(_precision) {}

    void reset() {
        std::fill(_histogram_A.begin(), _histogram_A.end(), 0u);
        std::fill(_histogram_B.begin(), _histogram_B.end(), 0u);
    }

    template <class DataVec> void stream_A(std::vector<DataVec> const& data) {
        for (auto& tv : data)
            for (auto byte : tv)
                _histogram_A[byte % _precision]++;
    }

    template <class DataVec> void stream_B(std::vector<DataVec> const& data) {
        for (auto& tv : data)
            for (auto byte : tv)
                _histogram_B[byte % _precision]++;
    }

    double compute_result() const;
};
