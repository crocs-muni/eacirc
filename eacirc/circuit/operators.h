#pragma once

#include "genotype.h"
#include <random>

namespace circuit {
class FnPool {
    std::array<Fn, static_cast<size_t>(Fn::_Size)> _pool;
    size_t _size;

public:
    FnPool() : _size(static_cast<size_t>(Fn::_Size)) {
        for (size_t i = 0; i != _size; ++i)
            _pool[i] = static_cast<Fn>(i);
    }

    size_t size() const { return _size; }
    Fn operator[](size_t i) const { return _pool[i]; }
};
} // namespace circuit

namespace circuit {
template <class IO, class Shape> struct Helper {
    template <class Generator> static u8 generate_argument(Generator& g) {
        std::uniform_int_distribution<u8> dst;
        return dst(g);
    }

    template <class Generator> static Fn generate_function(FnPool const& pool, Generator& g) {
        std::uniform_int_distribution<size_t> dst{0, pool.size() - 1};
        return pool[dst(g)];
    }

    template <class Generator>
    static Connectors<IO, Shape> generate_connectors(Generator& g, unsigned size) {
        using Type = typename Connectors<IO, Shape>::Type;

        std::uniform_int_distribution<Type> dst(0, (1u << size) - 1);
        return dst(g);
    }
};
} // namespace circuit

namespace circuit {
template <class IO, class Shape> struct Basic_Mutator {
private:
    FnPool const _fn_pool;
    unsigned _tv_size;

public:
    Basic_Mutator(unsigned tv_size) : _tv_size(tv_size) {}

    template <class Generator> void operator()(Genotype<IO, Shape>& circuit, Generator& g) {
        std::uniform_int_distribution<unsigned> node_dst{0, Genotype<IO, Shape>::num_of_nodes};

        circuit.node(node_dst(g)).function = Helper<IO, Shape>::generate_function(_fn_pool, g);
        circuit.node(node_dst(g)).argument = Helper<IO, Shape>::generate_argument(g);

        auto i = node_dst(g);
        if (i < Shape::x) {
            // mutate connectors in first layer
            std::uniform_int_distribution<unsigned> dst{0, _tv_size - 1};
            circuit.node(i).connectors.flip(dst(g));
        } else {
            // mutate connectors in other layers
            std::uniform_int_distribution<unsigned> dst{0, Shape::x - 1};
            circuit.node(i).connectors.flip(dst(g));
        }
    }
};

template <class IO, class Shape> struct Basic_Initializer {
private:
    FnPool const _fn_pool;
    unsigned _tv_size;

public:
    Basic_Initializer(unsigned tv_size) : _tv_size(tv_size) {}

    template <class Generator> void operator()(Genotype<IO, Shape>& circuit, Generator& g) {
        // for the first layer...
        for (unsigned i = 0; i != Shape::x; ++i) {
            auto node = circuit[0][i];

            node.connectors = (i < _tv_size) ? 1u << i : 0;
            node.function = Fn::XOR;
            node.argument = Helper<IO, Shape>::generate_argument(g);
        }

        // for the rest layers...
        for (unsigned i = 1; i != Shape::y; ++i)
            for (auto node : circuit[i]) {
                node.connectors = Helper<IO, Shape>::generate_connectors(g, Shape::x);
                node.function = Helper<IO, Shape>::generate_function(_fn_pool, g);
                node.argument = Helper<IO, Shape>::generate_argument(g);
            }
    }
};
} // namespace circuit
