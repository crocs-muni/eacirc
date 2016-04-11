#pragma once

#include <array>
#include <core/base.h>
#include <core/bitset.h>

namespace circuit {
    enum class Function : ui8 {
        NOP,
        CONS,
        AND,
        NAND,
        OR,
        XOR,
        NOR,
        NOT,
        SHIL,
        SHIR,
        ROTL,
        ROTR,
        MASK,
        _Size // this must be the last item in this enum
    };

    struct alignas(4) Node {
        Function function;
        ui8 argument;
        Bitset<16> connectors;
    };

    template <unsigned X, unsigned Y> struct Circuit {
        const static unsigned x = X;
        const static unsigned y = Y;

    private:
        std::array<std::array<Node, X>, Y> _nodes;

    public:
        std::array<Node, X>& layers() { return _nodes; }
        const std::array<Node, X>& layers() const { return _nodes; }

        unsigned node_count() { return x * y }
    };
}
