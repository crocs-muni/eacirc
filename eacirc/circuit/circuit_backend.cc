#include "circuit_backend.h"
#include "../solvers/criteria.h"
#include "../solvers/local_search.h"
#include "genetics.h"

namespace circuit {

template <typename Circuit, typename Sseq>
std::unique_ptr<solver> make_global_search(core::json const& config, Sseq& seed_source) {
    fn_set funcs(config["functions"]);
    std::size_t num_of_categories = config["num-of-categories"];
    std::size_t num_of_iterations = config["num-of-iterations"];

    auto mut = basic_mutator<Circuit>{funcs};
    auto eval = categories_evaluator<Circuit>{num_of_categories};
    auto init = basic_initializer<Circuit>{funcs};
    auto stop = max_iterations{num_of_iterations};

    return std::make_unique<
            local_search<Circuit, decltype(mut), decltype(eval), decltype(init), decltype(stop)>>(
            mut, eval, init, stop, seed_source);
}

circuit_backend::circuit_backend(core::json const& config,
                                 core::stream& stream_a,
                                 core::stream& stream_b,
                                 core::main_seed_source& seed_source)
    : backend(stream_a, stream_b) {
    if (config["solver"] == "global-search") {
        _solver = make_global_search<circuit<16, 1, 8, 5>>(config["global-search"], seed_source);
    }
}

double circuit_backend::train() {
    _solver->run();
    return _solver->replace_datasets(stream_a(), stream_b());
}

} // namespace circuit
