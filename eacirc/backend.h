#pragma once

#include <core/base.h>
#include <solvers/solver.h>

struct Backend {
    virtual ~Backend()= default;

    virtual std::unique_ptr<solvers::Solver> solver(u32 seed) const = 0;
};
