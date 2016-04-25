#pragma once

#include <cstddef>

namespace solvers {
template <class T, class S> struct Solution {
    T genotype;
    S score;
};

class Solver {
public:
    virtual ~Solver() = default;

    virtual void init() = 0;
    virtual void run(const size_t iterations) = 0;
    virtual double reevaluate() = 0;

    virtual void data(void* data) = 0;
};
} // namespace solvers
