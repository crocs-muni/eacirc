#pragma once

#include <core/base.h>
#include <core/dataset.h>
#include <cstddef>

template <class T, class S> struct Solution {
    T genotype;
    S score;
};

struct Solver {
    virtual ~Solver() = default;

    virtual void replace_datasets(Dataset const&, Dataset const&) = 0;

    virtual void init() = 0;
    virtual void run(const size_t iterations) = 0;
    virtual double reevaluate() = 0;
};
