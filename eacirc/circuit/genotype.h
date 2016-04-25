#pragma once

#include <array>
#include <core/base.h>
#include <core/bitset.h>
#include <core/compiletime.h>

namespace circuit {
enum class Fn : u8 {
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
    _Size // this must be the last item of this enum
};

template <class Def> using Connectors = Bitset<Max<Def::in, Def::x>::value>;

template <class Def> struct Node {
    Fn function;
    u8 argument;
    Connectors<Def> connectors;
};

template <class Def> class Genotype {
    using Layer = std::array<Node<Def>, Def::x>;
    using Storage = std::array<Layer, Def::y>;

    Storage _layers;

public:
    auto begin() -> typename Storage::iterator { return _layers.begin(); }
    auto begin() const -> typename Storage::const_iterator { return _layers.begin(); }

    auto end() -> typename Storage::iterator { return _layers.end(); }
    auto end() const -> typename Storage::const_iterator { return _layers.end(); }

    auto operator[](size_t i) -> Layer& { return _layers[i]; }
    auto operator[](size_t i) const -> Layer const& { return _layers[i]; }

    auto node(size_t i) -> Node<Def>& { return _layers.front()[i]; }
    auto node(size_t i) const -> Node<Def> const& { return _layers.front()[i]; }
};
} // namespace circuit
