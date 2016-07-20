#pragma once

#include "../ea-backend.h"
#include <ea-settings.h>
#include <ea-utils.h>

namespace ea {
namespace circuit {

struct circuit_factory : factory<backend> {
    circuit_factory(const settings &);

    virtual std::unique_ptr<backend> create() override;
};

} // namespace circuit
} // namespace ea
