#pragma once

#include "solvers/solver.h"
#include <core/dataset.h>

struct Backend {
    virtual ~Backend() = default;

    virtual std::unique_ptr<Solver> solver(u64 seed) = 0;
};
