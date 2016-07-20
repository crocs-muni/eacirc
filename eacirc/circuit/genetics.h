#pragma once

#include "circuit.h"

namespace ea {
namespace circuit {

template <class Def> struct basic_initializer {
    template <class Generator> void apply(circuit<Def> &, Generator &);
};

template <class Def> struct basic_mutator {
    template <class Generator> void apply(circuit<Def> &, Generator &);
};

} // namespace circuit
} // namespace ea
