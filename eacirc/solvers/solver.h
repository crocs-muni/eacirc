#pragma once

#include <core/stream.h>
#include <cstddef>

struct solver {
    virtual ~solver() = default;
    virtual void run() = 0;
    virtual double reevaluate() = 0;
    virtual double replace_datasets(core::stream& stream_a, core::stream& stream_b) = 0;
};
