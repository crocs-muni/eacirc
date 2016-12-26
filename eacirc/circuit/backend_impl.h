#pragma once

#include "backend.h"
#include "circuit.h"
#include "genetics.h"
#include <core/memory.h>
#include <fstream>
#include "solvers/solvers.h"
#include "solvers/solver_factory.h"

namespace circuit {

    template <typename Circuit> struct global_search : backend {
        template <typename Sseq>
        global_search(unsigned tv_size, json const& config, Sseq&& seed)
            : _function_set(config.at("function-set"))
            , _num_of_generations(config.at("num-of-generations"))
            , _solver(solvers::create_solver<Circuit, Sseq>(tv_size, config, std::forward<Sseq>(seed)))
              /*_solver(Circuit(tv_size),
                      ini(config.at("initializer"), _function_set),
                      mut(config.at("mutator"), _function_set),
                      eva(config.at("evaluator")),
                      std::forward<Sseq>(seed))*/ {}

        ~global_search() {
            {
                std::ofstream out("scores.txt");
                for (double score : _solver->scores())
                    out << score << std::endl;
            }
        }

        void train(dataset const& a, dataset const& b) override {
            _solver->reevaluate(a, b);
            _solver->run(_num_of_generations);
        }

        double test(dataset const& a, dataset const& b) override {
            return _solver->reevaluate(a, b);
        }

    private:
        using ini = basic_initializer;
        using mut = basic_mutator;
        using eva = categories_evaluator<Circuit>;

        fn_set _function_set;
        std::uint64_t _num_of_generations;
        std::unique_ptr<solvers::solver> _solver;
    };

    std::unique_ptr<backend>
    create_backend(unsigned tv_size, json const& config, default_seed_source& seed) {
        json const solver = config.at("solver");
        std::string solver_type = solver.at("type");

        if (solver_type == "global-search")
            return std::make_unique<global_search<circuit<8, 5, 1>>>(tv_size, solver, seed);
        else
            throw std::runtime_error("no such solver named [" + solver_type + "] is avalable");
    }

} // namespace circuit
