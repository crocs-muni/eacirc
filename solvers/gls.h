#pragma once

#include "individual.h"
#include "solvers.h"

namespace solvers {

    template <typename Genotype,
              typename Initializer,
              typename Mutator,
              typename Evaluator,
              typename Generator = default_random_generator>
    struct gls : solver {
        template <typename Sseq>
        gls(Genotype&& gen, Initializer&& ini, Mutator&& mut, std::unique_ptr<Evaluator> eva, Sseq&& seed)
            : _solution(std::move(gen))
            , _neighbour(_solution)
            , _initializer(std::move(ini))
            , _mutator(std::move(mut))
            , _basic_evaluator(std::move(eva))
            , _generator(std::forward<Sseq>(seed)) {
            _initializer.apply(_solution.genotype, _generator);
        }

        double run(std::uint64_t generations) {
            for (std::uint64_t i = 0; i != generations; ++i)
                _step();
            dump_to_graph("circuit.dot");
            return _solution.score;
        }

        double reevaluate(dataset const& a, dataset const& b) {
            _basic_evaluator->change_datasets(a, b);
            _solution.score = _basic_evaluator->apply(_solution.genotype);
            _scores.emplace_back(_solution.score);
            return _solution.score;
        }

        auto scores() const -> view<std::vector<double>::const_iterator> {
            return make_view(_scores);
        }

        void dump_to_graph(const std::string &filename) {
            _solution.genotype.dump_to_graph(filename);
            auto copy = _solution;
            copy.genotype.prune();
            copy.genotype.dump_to_graph("pruned_" + filename);
        }


    private:
        individual<Genotype, double> _solution;
        individual<Genotype, double> _neighbour;

        Initializer _initializer;
        Mutator _mutator;
        std::unique_ptr<Evaluator> _basic_evaluator;
        Generator _generator;

        std::vector<double> _scores;

        void _step() {
            _neighbour = _solution;
            _mutator.apply(_neighbour.genotype, _generator);

            _neighbour.score = _basic_evaluator->apply(_neighbour.genotype);
            if (std::abs(_neighbour.score - _solution.score) < 0.00001) {
                if (_neighbour.score == 0 && _solution.genotype.count_connectors() <= _neighbour.genotype.count_connectors())
                    _solution = std::move(_neighbour);
                else if (_neighbour.score > 0.9 && _solution.genotype.count_connectors() >= _neighbour.genotype.count_connectors())
                    _solution = std::move(_neighbour);

            } else if (_solution <= _neighbour) {
                _solution = std::move(_neighbour);
            }
            _scores.emplace_back(_solution.score);
        }
    };

} // namespace solvers
