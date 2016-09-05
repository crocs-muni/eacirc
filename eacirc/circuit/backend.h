#pragma once

#include "../backend.h"
#include <core/json.h>
#include <core/random.h>
#include <memory>

namespace circuit {

    std::unique_ptr<backend>
    create_backend(unsigned tv_size, json const& config, default_seed_source& seed);

} // namespace circuit
