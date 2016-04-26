#pragma once

#include "base.h"
#include <vector>

template <unsigned Size> struct Dataset {
    Dataset(size_t size) : stream_A(size), stream_B(size) {}

    std::vector<DataVec<Size>> stream_A;
    std::vector<DataVec<Size>> stream_B;
};
