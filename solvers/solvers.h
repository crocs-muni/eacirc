#pragma once

#include "individual.h"
#include "local_search.h"
#include "simulated_annealing.h"
#include <core/dataset.h>
#include <core/random.h>
#include "../eacirc/circuit/backend.h"
#include "../eacirc/circuit/circuit.h"
#include "../eacirc/circuit/genetics.h"

namespace solvers {

    template <typename Genotype,
              typename Initializer,
              typename Mutator,
              typename Evaluator,
              typename Generator = default_random_generator>
    struct solver {

        virtual ~solver() = default;

        virtual double run(std::uint64_t generations = 1) = 0;

        virtual double reevaluate(dataset const& a, dataset const& b) = 0;

        auto scores() const -> view<std::vector<double>::const_iterator> {
            return make_view(_scores);
        }

    protected:
        template <typename Sseq>
        solver(Mutator&& mut, Evaluator&& eva, Sseq&& seed)
            : _mutator(std::move(mut))
            , _evaluator(std::move(eva))
            , _generator(std::forward<Sseq>(seed)) {}

        Mutator mutator() const { return _mutator; }
        Evaluator evaluator() const { return _evaluator; }
        Generator generator() const { return _generator; }
        std::vector<double> scores() const { return _scores; }

    private:
        Mutator _mutator;
        Evaluator _evaluator;
        Generator _generator;

        std::vector<double> _scores;
    };

    template <typename Circuit>
    std::unique_ptr<solver>
    create_solver(unsigned tv_size, const json& config, default_seed_source& seed) {
        using ini = circuit::basic_initializer;
        using mut = circuit::basic_mutator;
        using eva = circuit::categories_evaluator<Circuit>;

        circuit::fn_set function_set(config.at("function-set"));

        std::string solver = config.at("solver");

        if (solver == "local-search")
            return std::make_unique<local_search<Circuit, ini, mut, eva>>(Circuit(tv_size),
                                                                          ini(config.at("initializer"), function_set),
                                                                          mut(config.at("mutator"), function_set),
                                                                          eva(config.at("evaluator")),
                                                                          std::forward<Sseq>(seed));
        if (solver == "simulated-annealing")
            ;
        else
            throw std::runtime_error("no such solver named [" + solver + "] is avalable");
    }
}
