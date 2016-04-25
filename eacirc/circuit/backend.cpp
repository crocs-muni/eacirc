#include "backend.h"
#include "circuit.h"
#include "genotype.h"
#include "operators.h"
#include <core/base.h>
#include <solvers/local_search.h>

using namespace solvers;

namespace circuit {
using Def = Circuit<16, 1, 8, 5>;

template <class T>
using BasicSolver =
        LocalSearch<Genotype<T>, basic_mutator<T>, categories_evaluator<T>, basic_initializer<T>>;

std::unique_ptr<Solver> Backend::solver(u32 seed) const {
    FnPool fn_pool;

    // clang-format off
        return std::make_unique<BasicSolver<Def>>(
                basic_mutator<Def>{fn_pool},
                categories_evaluator<Def>{8},
                basic_initializer<Def>{fn_pool}, seed);
    // clang-format on
}
} // namespace circuit
