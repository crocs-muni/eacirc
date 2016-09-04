#pragma once

#include "individual.h"
#include <core/dataset.h>
#include <core/random.h>

namespace solvers {

    template <typename Genotype,
              typename Initializer,
              typename Mutator,
              typename Evaluator,
              typename Generator = default_random_generator>
    struct local_search {
        template <typename Sseq>
        local_search(Genotype&& gen, Initializer&& ini, Mutator&& mut, Evaluator&& eva, Sseq&& seed)
            : _solution(std::move(gen))
            , _neighbour(_solution)
            , _initializer(std::move(ini))
            , _mutator(std::move(mut))
            , _evaluator(std::move(eva))
            , _generator(std::forward<Sseq>(seed)) {
            _initializer.apply(_solution.genotype, _generator);
        }

        double run(std::uint64_t generations) {
            for (std::uint64_t i = 0; i != generations; ++i)
                _step();
            return _solution.score;
        }

        double reevaluate(dataset const& a, dataset const& b) {
            _evaluator.change_datasets(a, b);
            return _solution.score = _evaluator.apply(_solution.genotype);
        }

    private:
        individual<Genotype, double> _solution;
        individual<Genotype, double> _neighbour;

        Initializer _initializer;
        Mutator _mutator;
        Evaluator _evaluator;
        Generator _generator;

        void _step() {
            _neighbour = _solution;
            _mutator.apply(_neighbour.genotype, _generator);

            _neighbour.score = _evaluator.apply(_neighbour.genotype);
            if (_solution <= _neighbour) {
                using std::swap;
                swap(_solution, _neighbour);
            }
        }
    };

} // namespace solvers
