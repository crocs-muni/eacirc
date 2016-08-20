#pragma once

#include "solver.h"
#include <core/random.h>
#include <utility>

template <typename Type,
          typename Mut,
          typename Eval,
          typename Init,
          typename Stop,
          typename Gen = core::default_rng>
struct local_search : solver {
    template <typename Sseq>
    local_search(Mut mutator, Eval evaluator, Init initializer, Stop stopping, Sseq& seed_source)
        : _mutator(std::move(mutator))
        , _evaluator(std::move(evaluator))
        , _initializer(std::move(initializer))
        , _generator(seed_source)
        , _stopping(stopping) {
        _initializer.apply(_solution_a, _generator);
        _score_a = _evaluator.apply(_solution_a);
    }

    void run() override {
        for (Stop criterion(_stopping); criterion(_solution_a); ++criterion)
            _step();
    }

    double reevaluate() override {
        return _score_a = _evaluator.apply(_solution_a);
    }

    double replace_datasets(core::stream& stream_a, core::stream& stream_b) override {
        _evaluator.replace_datasets(stream_a, stream_b);
        return reevaluate();
    }

private:
    Type _solution_a;
    Type _solution_b;

    double _score_a;
    double _score_b;

    Mut _mutator;
    Eval _evaluator;
    Init _initializer;
    Gen _generator;
    Stop _stopping;

    void _step() {
        _solution_b = _solution_a;
        _mutator.apply(_solution_b, _generator);

        _score_b = _evaluator.apply(_solution_b);
        if (_score_a <= _score_b) {
            using std::swap;
            swap(_solution_a, _solution_b);
            swap(_score_a, _score_b);
        }
    }
};
