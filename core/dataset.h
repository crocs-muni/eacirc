#pragma once

#include "base.h"
#include <vector>

template <unsigned Size> struct Dataset {
    std::vector<DataVec<Size>> stream_A;
    std::vector<DataVec<Size>> stream_B;
};
