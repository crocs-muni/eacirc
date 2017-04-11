#pragma once

#include "../backend.h"
#include <eacirc-core/json.h>
#include <eacirc-core/random.h>
#include <memory>

namespace circuit {

    std::unique_ptr<backend>
    create_backend(unsigned tv_size, json const& config, default_seed_source& seed);

} // namespace circuit
