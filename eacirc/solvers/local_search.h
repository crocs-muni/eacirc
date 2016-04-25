#pragma once

#include "solver.h"
#include <core/base.h>
#include <random>
#include <core/project.h>

namespace solvers {
template <
        class Genotype, class Mutator, class Evaluator, class Initializer,
        class Generator = std::mt19937>
class LocalSearch : public Solver {
    Mutator _mutator;
    Evaluator _evaluator;
    Initializer _initializer;

    Generator _generator;
    Solution<Genotype, double> _solution;

public:
    LocalSearch(Mutator&& mutator, Evaluator&& evaluator, Initializer&& inititalizer, u32 seed)
        : _mutator(std::move(mutator))
        , _evaluator(std::move(evaluator))
        , _initializer(std::move(inititalizer))
        , _generator(seed) {}

    void data(void* data) override { _evaluator.data(reinterpret_cast<Data<16, 1>*>(data)); }

    void init() override { _initializer(_solution.genotype, _generator); }

    void run(const size_t iterations) override {
        for (size_t i = 0; i != iterations; ++i) {
            auto neighbour = _solution;

            _mutator(neighbour.genotype, _generator);
            neighbour.score = _evaluator(neighbour.genotype);

            if (_solution.score <= neighbour.score)
                _solution = std::move(neighbour);
        }
    }

    double reevaluate() override { return _solution.score = _evaluator(_solution.genotype); }
};
} // namespace solvers
