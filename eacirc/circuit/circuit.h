#pragma once

namespace circuit {
template <unsigned In, unsigned Out, unsigned X, unsigned Y> struct Circuit {
    const static unsigned in = In;
    const static unsigned out = Out;
    const static unsigned x = X;
    const static unsigned y = Y;
};

template <class Def> struct CircuitTraits {
    const static unsigned num_of_nodes = Def::x * (Def::y - 1) + Def::out - 1;
};
} // namespace circuit
