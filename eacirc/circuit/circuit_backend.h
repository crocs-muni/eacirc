#pragma once

#include "../backend.h"
#include "../solvers/solver.h"
#include <core/json.h>
#include <core/random.h>
#include <memory>

namespace circuit {

struct circuit_backend : backend {
    circuit_backend(core::json const& config,
                    core::stream& stream_a,
                    core::stream& stream_b,
                    core::main_seed_source& seed_source);

    double train() override;

private:
    std::unique_ptr<solver> _solver;
};

} // namespace circuit
