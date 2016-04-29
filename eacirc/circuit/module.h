#pragma once

#include "backend.h"

namespace circuit {
struct Module {
    static std::unique_ptr<Backend> get_backend();
};
} // namespace circuit
