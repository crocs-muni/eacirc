#pragma once

#include "ea-utils.h"

namespace ea {

struct backend {
    virtual ~backend();
};

struct backend_factory : factory<backend> {
    std::unique_ptr<backend> create() override;
};

} // namespace ea
