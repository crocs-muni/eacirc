#pragma once

#include "evaluators/categories.h"
#include "interpreter.h"
#include <core/dataset.h>

namespace circuit {
template <class IO, class Shape> struct Categories_Evaluator {
private:
    Categories _categories;

    Dataset inA, inB;
    Dataset outA, outB;

public:
    Categories_Evaluator(unsigned precision) : _categories(precision) {}

    void replace_datasets(Dataset const& a, Dataset const& b) {
        inA = a;
        inB = b;

        outA = Dataset(IO::out, a.num_of_tvs());
        outB = Dataset(IO::out, b.num_of_tvs());
    }

    double operator()(Genotype<IO, Shape> const& circuit) {
        Interpreter<IO, Shape, std::array> interpreter(circuit);

        discard_iterator<Dataset::Vec> discard;

        std::transform(inA.begin(), inA.end(), outA.begin(), discard, interpreter);
        std::transform(inB.begin(), inB.end(), outB.begin(), discard, interpreter);

        _categories.reset();
        _categories.stream_A(outA);
        _categories.stream_B(outB);
        return _categories.compute_result();
    }
};
} // namespace circuit
