#pragma once

#include "../backend.h"
#include <memory>

namespace circuit {
class Backend : public ::Backend {
public:
    std::unique_ptr<solvers::Solver> solver(u32 seed) const override;
};
} // namespace circuit
