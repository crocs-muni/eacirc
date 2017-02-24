#pragma once

#include "individual.h"
#include "solvers.h"
#include <core/dataset.h>
#include <core/random.h>
#include <core/view.h>
#include <eacirc/circuit/genetics.h>

namespace solvers {

    template <typename Genotype,
              typename Initializer,
              typename Mutator,
              typename Evaluator,
              typename Generator = default_random_generator>
    struct vns : solver {
        template <typename Sseq>
        vns(Genotype&& gen,
            Initializer&& ini,
            Mutator&& mut,
            std::unique_ptr<Evaluator> eva,
            Sseq&& seed)
            : _solution(std::move(gen))
            , _neighbour(_solution)
            , _initializer(std::move(ini))
            , _basic_mutator(std::move(mut))
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
            _evaluator->change_datasets(a, b);
            _solution.score = _evaluator->apply(_solution.genotype);
            _scores.emplace_back(_solution.score);
            return _solution.score;
        }

        auto scores() const -> view<std::vector<double>::const_iterator> {
            return make_view(_scores);
        }

    private:
        individual<Genotype, double> _solution;
        individual<Genotype, double> _neighbour;

        Initializer _initializer;
        Mutator _basic_mutator;

        std::unique_ptr<Evaluator> _evaluator;
        Generator _generator;

        std::vector<double> _scores;

        void _step() {
            _neighbour = _solution;
            _basic_mutator.apply(_neighbour.genotype, _generator);

            _neighbour.score = _evaluator->apply(_neighbour.genotype);

            /**
             * This is not exactly Variable neighbourhood search, but it more reflects problem with
             * zero fitness or overconnected solutions in EACirc
             *
             * True VNS would define various neigbourhoods and in case of local optima
             * it would change between neighbourhoods to escape from it.
             */
            if (_neighbour.score == 0) {
                std::size_t i = 0;
                while (i < 10 && _neighbour.score == 0) {
                    circuit::extend<Genotype, Generator>(_neighbour.genotype, _generator, 2);
                    _neighbour.score = _evaluator->apply(_neighbour.genotype);
                    ++i;
                }
            } else if (_neighbour.score > 0.9 && _solution.score > 0.9) {
                circuit::reduce<Genotype, Generator>(_neighbour.genotype, _generator, 2);
            }

            // we need to reevaluate the solution after the changes
            _neighbour.score = _evaluator->apply(_neighbour.genotype);

            if (_solution <= _neighbour)
                _solution = std::move(_neighbour);

            _scores.emplace_back(_solution.score);
        }
    };

} // namespace solvers
