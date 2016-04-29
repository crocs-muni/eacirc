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

template <class IO, class Shape> using Connectors = Bitset<Max<IO::in, Shape::x>::value>;

template <class IO, class Shape> struct Node {
    Fn function = Fn::NOP;
    u8 argument = 0u;
    Connectors<IO, Shape> connectors = 0u;
};

template <class IO, class Shape> struct Genotype {
private:
    using Layer = std::array<Node<IO, Shape>, Shape::x>;
    using Storage = std::array<Layer, Shape::y>;

    Storage _layers;

public:
    const static unsigned num_of_nodes = Shape::x * (Shape::y - 1) + IO::out - 1;

    auto begin() -> typename Storage::iterator { return _layers.begin(); }
    auto begin() const -> typename Storage::const_iterator { return _layers.begin(); }

    auto end() -> typename Storage::iterator { return _layers.end(); }
    auto end() const -> typename Storage::const_iterator { return _layers.end(); }

    auto operator[](size_t i) -> Layer& { return _layers[i]; }
    auto operator[](size_t i) const -> Layer const& { return _layers[i]; }

    auto node(size_t i) -> Node<IO, Shape>& { return _layers.front()[i]; }
    auto node(size_t i) const -> Node<IO, Shape> const& { return _layers.front()[i]; }
};
} // namespace circuit
