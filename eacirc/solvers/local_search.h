#pragma once

#include <cstddef>
#include <utility>

template <class Genotype, class Score> struct solution {
    Genotype genotype;
    Score score;
};

template <class Type, class Initializer, class Mutator, class Evaluator>
struct local_search {
    local_search(Initializer &&initializer, Mutator &&mutator,
                 Evaluator &&evaluator)
        : _initializer(initializer)
        , _mutator(mutator)
        , _evaluator(evaluator) {
        _initializer.apply(_solution.genotype);
    }

    void run(const std::size_t iterations) {
        for (size_t i = 0; i != iterations; ++i) {
            auto neighbour = _solution;

            _mutator.apply(neighbour.genotype);
            neighbour.score = _evaluator.apply(neighbour.genotype);

            if (_solution.score <= neighbour.score)
                _solution = std::move(neighbour);
        }
    }

private:
    Initializer _initializer;
    Mutator _mutator;
    Evaluator _evaluator;
    solution<Type, double> _solution;
};
