#pragma once

#include "backend.h"
#include "evaluators.h"
#include "genotype.h"
#include "operators.h"
#include <solvers/local_search.h>

namespace circuit {
template <unsigned I, unsigned O> struct IO {
    const static unsigned in = I;
    const static unsigned out = O;
};

template <unsigned X, unsigned Y> struct Shape {
    const static unsigned x = X;
    const static unsigned y = Y;
};

template <class IO, class Shape> struct Circuit : Backend {
private:
    using Mutator = Basic_Mutator<IO, Shape>;
    using Evaluator = Categories_Evaluator<IO, Shape>;
    using Initializer = Basic_Initializer<IO, Shape>;

    using CircuitSolver = LocalSearch<Genotype<IO, Shape>, Initializer, Mutator, Evaluator>;

    unsigned _tv_size;
    unsigned _precison;

public:
    Circuit(unsigned tv_size, unsigned precison) : _tv_size(tv_size), _precison(precison) {}

    std::unique_ptr<Solver> solver(u64 seed) final {
        return std::make_unique<CircuitSolver>(
                Initializer{_tv_size}, Mutator{_tv_size}, Evaluator{_precison}, seed);
    }
};
} // namespace circuit
