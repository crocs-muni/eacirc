#pragma once

#include "backend.h"
#include "circuit.h"
#include "genetics.h"
#include <solvers/local_search.h>

namespace circuit {

    template <typename Circuit> struct global_search : backend {
        template <typename Sseq>
        global_search(unsigned tv_size, json const& config, Sseq&& seed)
            : _function_set(config["function-set"])
            , _num_of_generations(config["num-of-generations"])
            , _solver(Circuit(tv_size),
                      ini(config["initializer"], _function_set),
                      mut(config["mutator"], _function_set),
                      eva(config["evaluator"]),
                      std::forward<Sseq>(seed)) {}

        void train(dataset const& a, dataset const& b) override {
            _solver.reevaluate(a, b);
            _solver.run(_num_of_generations);
        }

        double test(dataset const& a, dataset const& b) override {
            return _solver.reevaluate(a, b);
        }

    private:
        using ini = basic_initializer;
        using mut = basic_mutator;
        using eva = categories_evaluator<Circuit>;

        fn_set _function_set;
        std::uint64_t _num_of_generations;
        solvers::local_search<Circuit, ini, mut, eva> _solver;
    };

    std::unique_ptr<backend>
    create_backend(unsigned tv_size, json const& config, default_seed_source& seed) {
        return std::make_unique<global_search<circuit<8, 5, 1>>>(tv_size, config, seed);
    }

} // namespace circuit
