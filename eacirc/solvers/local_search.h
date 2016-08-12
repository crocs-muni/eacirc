#pragma once

#include <cstddef>
#include <utility>

template <class Type, class Mut, class Eval, class Init> struct local_search {
    local_search(Mut &mutator, Eval &evaluator, Init &initializer)
        : _mutator(mutator)
        , _evaluator(evaluator)
        , _initializer(initializer) {
        _initializer.apply(_solution_a);
        _score_a = _evaluator(_solution_a);
    }

    void run(const std::size_t iterations) {
        for (std::size_t i = 0; i != iterations; ++i)
            _step();
    }

    double reevaluate() { return _score_a = _evaluator.apply(_solution_a); }

private:
    Type _solution_a;
    Type _solution_b;

    double _score_a;
    double _score_b;

    Mut &_mutator;
    Eval &_evaluator;
    Init &_initializer;

    void _step() {
        _solution_b = _solution_a;
        _mutator.apply(_solution_b);

        _score_b = _evaluator.apply(_solution_b);
        if (_score_a <= _score_b) {
            using std::swap;
            swap(_solution_a, _solution_b);
            swap(_score_a, _score_b);
        }
    }
};
