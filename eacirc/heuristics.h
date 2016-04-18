#pragma once

#include <core/base.h>
#include <random>

struct Heuristic {
    virtual ~Heuristic() = default;

    virtual double reevaluate() = 0;
    virtual double init() = 0;

    virtual void step() = 0;
};

template <
        class Genotype, class Mutator, class Evaluator, class Initializer,
        class Generator = std::mt19937>
class LocalSearch : public Heuristic {
    Mutator _mutator;
    Genotype _genotype;
    Evaluator _evaluator;
    Generator _generator;
    Initializer _initializer;
    double _score;

public:
    LocalSearch(const u32 seed) : _generator(seed) {}

    double reevaluate() override { return _score = _evaluator(_genotype); }

    double init() override {
        _initializer(_genotype, _generator);
        return reevaluate();
    }
    void step() override {
        const auto neighbour_genotype = _mutator(_genotype, _generator);
        const auto neighbour_score = _evaluator(neighbour_genotype);

        if (_score < neighbour_score) {
            _genotype = std::move(neighbour_genotype);
            _score = neighbour_score;
        }
    }
};
