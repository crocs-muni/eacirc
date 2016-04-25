#pragma once

#include "../evaluators/categories.h"
#include "circuit.h"
#include "genotype.h"
#include "interpreter.h"
#include <core/project.h>
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

template <class Def> struct Helper {
    template <class Generator> static u8 generate_argument(Generator& g) {
        std::uniform_int_distribution<u8> dst;
        return dst(g);
    }

    template <class Generator> static Fn generate_function(FnPool const& pool, Generator& g) {
        std::uniform_int_distribution<size_t> dst{0, pool.size() - 1};
        return pool[dst(g)];
    }

    template <class Generator>
    static Connectors<Def> generate_connectors(Generator& g, unsigned size) {
        std::uniform_int_distribution<typename Connectors<Def>::Type> dst(0, (1u << size) - 1);
        return dst(g);
    }
};

template <class Def> class basic_mutator {
    FnPool const _fn_pool;

public:
    basic_mutator(FnPool const& fn_pool) : _fn_pool(fn_pool) {}

    template <class Generator> void operator()(Genotype<Def>& circ, Generator& g) {
        std::uniform_int_distribution<unsigned> node_dst{0, CircuitTraits<Def>::num_of_nodes};

        circ.node(node_dst(g)).function = Helper<Def>::generate_function(_fn_pool, g);
        circ.node(node_dst(g)).argument = Helper<Def>::generate_argument(g);

        auto i = node_dst(g);
        if (i < Def::x) {
            // mutate connectors in first layer
            std::uniform_int_distribution<unsigned> dst{0, Def::in - 1};
            circ.node(i).connectors.flip(dst(g));
        } else {
            // mutate connectors in other layers
            std::uniform_int_distribution<unsigned> dst{0, Def::x - 1};
            circ.node(i).connectors.flip(dst(g));
        }
    }
};

template <class Def> class basic_initializer {
    FnPool const _fn_pool;

public:
    basic_initializer(FnPool const& fn_pool) : _fn_pool(fn_pool) {}

    template <class Generator> void operator()(Genotype<Def>& circ, Generator& g) {
        // for the first layer
        for (unsigned i = 0; i != Def::x; ++i) {
            auto node = circ[0][i];

            node.connectors = 1u << i;
            node.function = Fn::XOR;
            node.argument = Helper<Def>::generate_argument(g);
        }

        // for the rest layers
        for (unsigned i = 1; i != Def::y; ++i)
            for (auto node : circ[i]) {
                node.connectors = Helper<Def>::generate_connectors(g, Def::x);
                node.function = Helper<Def>::generate_function(_fn_pool, g);
                node.argument = Helper<Def>::generate_argument(g);
            }
    }
};

template <class Def> class categories_evaluator : public Evaluator<Def::in, Def::out> {
    Categories _categories;

public:
    categories_evaluator(unsigned precision)
        : _categories(precision) {}

    double operator()(Genotype<Def> const& circuit) {
        Interpreter<Def, std::array> interpreter(circuit);
        std::transform(this->_data->ins_A.begin(), this->_data->ins_A.end(), this->_data->outs_A.begin(), interpreter);
        std::transform(this->_data->ins_B.begin(), this->_data->ins_B.end(), this->_data->outs_B.begin(), interpreter);

        _categories.reset();
        _categories.stream_A(this->_data->outs_A);
        _categories.stream_B(this->_data->outs_B);
        return _categories.compute_result();
    }
};
} // namespace circuit
