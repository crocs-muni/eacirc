#pragma once

#include "functions.h"
#include <array>
#include <core/bitset.h>
#include <core/traits.h>

namespace circuit {

template <unsigned I, unsigned O, unsigned X, unsigned Y> struct circuit {
    static constexpr unsigned in = I;
    static constexpr unsigned out = O;
    static constexpr unsigned x = X;
    static constexpr unsigned y = Y;
    static constexpr unsigned num_of_nodes = x * (y - 1) + out - 1;

    using connectors = core::bitset<core::max<in, x>::value>;

    struct node {
        fn function{fn::NOP};
        std::uint8_t argument{0u};
        connectors connectors{0u};
    };

    using layer = std::array<node, x>;
    using layout = std::array<layer, y>;

    typename layout::iterator begin() {
        return _layers.begin();
    }

    typename layout::iterator end() {
        return _layers.end();
    }

    typename layout::const_iterator begin() const {
        return _layers.begin();
    }

    typename layout::const_iterator end() const {
        return _layers.end();
    }

    layer& operator[](std::size_t i) {
        return _layers[i];
    }

    const layer& operator[](std::size_t i) const {
        return _layers[i];
    }

private:
    layout _layers;
};

} // namespace circuit
