#pragma once

#include <cstddef>

template <class G, class S> struct solution {
    G genotype = G();
    S score = S();
};

#include <random>
#include <utility>

template <class T> struct local_search {
    local_search(std::uint64_t seed)
        : _generator(seed) {
        _initializer(_solution.genotype, _generator);
    }

    void run(const std::size_t iterations) {
        for (size_t i = 0; i != iterations; ++i) {
            auto neighbour = _solution;

            _mutator(neighbour.genotype, _generator);
            neighbour.score = _evaluator(neighbour.genotype);

            if (_solution.score <= neighbour.score)
                _solution = std::move(neighbour);
        }
    }

private:
    std::mt19937 _generator;
    solution<T, double> _solution;
};
