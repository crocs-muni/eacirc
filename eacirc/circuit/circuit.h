#pragma once

#include "../backend.h"
#include <array>
#include <core/bitset.h>
#include <core/traits.h>

namespace circuit {

template <unsigned In, unsigned Out, unsigned X, unsigned Y> struct def {
    constexpr static unsigned in = In;
    constexpr static unsigned out = Out;
    constexpr static unsigned x = X;
    constexpr static unsigned y = Y;
};

enum class function {
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

template <class Def>
using connectors = core::bitset<core::max<Def::in, Def::x>::value>;

template <class Def> struct node {
    node()
        : fn(function::NOP)
        , arg(0u)
        , conns(0u) {}

    function fn;
    std::uint8_t arg;
    connectors<Def> conns;
};

template <class Def> struct circuit final : backend {
    using node = node<Def>;
    using layer = std::array<node, Def::x>;
    using layout = std::array<layer, Def::y>;

    typename layout::iterator begin() { return _layers.begin(); }
    typename layout::iterator end() { return _layers.end(); }

    typename layout::const_iterator begin() const { return _layers.begin(); }
    typename layout::const_iterator end() const { return _layers.end(); }

    layer &operator[](std::size_t i) { return _layers[i]; }
    const layer &operator[](std::size_t i) const { return _layers[i]; }

    constexpr static std::size_t num_of_nodes
            = Def::x * (Def::y - 1) + Def::out - 1;

private:
    layout _layers;
};

} // namespace circuit
