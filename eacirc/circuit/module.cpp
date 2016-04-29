#include "module.h"
#include "circuit.h"

namespace circuit {
std::unique_ptr<Backend> Module::get_backend() {
    return std::make_unique<Circuit<IO<16, 1>, Shape<8, 5>>>(16, 8);
};
} // namespace circuit
