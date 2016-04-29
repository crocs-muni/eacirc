#pragma once

#include "solver.h"
#include <core/base.h>
#include <random>

template <
        class Genotype, class Initializer, class Mutator, class Evaluator,
        class Generator = std::mt19937>
struct LocalSearch : Solver {
private:
    Mutator _mutator;
    Evaluator _evaluator;
    Initializer _initializer;

    Generator _generator;
    Solution<Genotype, double> _solution;

public:
    LocalSearch(Initializer&& inititalizer, Mutator&& mutator, Evaluator&& evaluator, u64 seed)
        : _mutator(std::move(mutator))
        , _evaluator(std::move(evaluator))
        , _initializer(std::move(inititalizer))
        , _generator(seed) {}

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

    void replace_datasets(Dataset const& a, Dataset const& b) override {
        _evaluator.replace_datasets(a, b);
    }
};
