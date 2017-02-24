#pragma once

#include "connectors.h"
#include "functions.h"
#include <core/vec.h>

namespace circuit {

    template <unsigned DimX, unsigned DimY, unsigned Out> struct circuit {
        static constexpr unsigned x = DimX;
        static constexpr unsigned y = DimY;
        static constexpr unsigned output_size = Out;

        using output = vec<Out>;
        using connectors_type = connectors<32>;

        struct node {
            fn function{fn::NOP};
            std::uint8_t argument{0u};
            connectors_type connectors{0u};
        };

        using layer = std::array<node, x>;
        using layers = std::array<layer, y>;

        using iterator = typename layers::iterator;
        using const_iterator = typename layers::const_iterator;

        circuit(unsigned input)
            : _input(input) {}

        circuit(circuit&&) = default;
        circuit(circuit const&) = default;

        circuit& operator=(circuit&&) = default;
        circuit& operator=(circuit const&) = default;

        unsigned input() const { return _input; }

        iterator begin() { return _layers.begin(); }
        const_iterator begin() const { return _layers.begin(); }

        iterator end() { return _layers.end(); }
        const_iterator end() const { return _layers.end(); }

        layer& operator[](std::size_t const i) {
            ASSERT(i < DimY);
            return _layers[i];
        }

        layer const& operator[](std::size_t const i) const {
            ASSERT(i < DimY);
            return _layers[i];
        }

        std::size_t count_connectors() {
            std::size_t count = 0;
            for (auto l : _layers)
                for (auto c : l)
                    count += c.connectors.count_connectors();
            return count;
        }

    private:
        layers _layers;
        unsigned _input;
    };

} // namespace circuit
