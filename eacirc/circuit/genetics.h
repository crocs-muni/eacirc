#pragma once

#include "../dataset.h"
#include "../statistics.h"
#include "circuit.h"
#include "interpreter.h"
#include <core/random.h>

namespace circuit {

template <class Def> struct helper {
    template <class Generator> static std::uint8_t generate_argument(Generator &g) {
        std::uniform_int_distribution<std::uint8_t> dst;
        return dst(g);
    }

    template <class Generator> static connectors<Def> generate_connetors(Generator &g, std::size_t size) {
        std::uniform_int_distribution<connectors<Def>> dst{0, (1u << size) - 1};
        return dst(g);
    }
};

template <class Def> struct basic_mutator {
    basic_mutator(const sample_pool<function> &pool)
        : changes_of_functions(2)
        , changes_of_arguments(2)
        , changes_of_connectors(3)
        , function_generator{pool} {
    }

    template <class Generator> void apply(circuit<Def> &circuit, Generator &g) const {
        std::uniform_int_distribution<std::size_t> x{0, Def::x};
        std::uniform_int_distribution<std::size_t> y{0, Def::y};

        // mutate functions
        for (size_t i = 0; i != changes_of_functions; ++i) {
            circuit[y(g)][x(g)].function = function_generator(g);
        }

        // mutate arguments
        for (size_t i = 0; i != changes_of_arguments; ++i) {
            circuit[y(g)][x(g)].argument = helper<Def>::generate_argument(g);
        }

        // mutate connectors
        for (size_t i = 0; i != changes_of_connectors; ++i) {
            std::uniform_int_distribution<std::size_t> dst;
            std::uniform_int_distribution<std::size_t>::param_type first_layer{0, Def::in};
            std::uniform_int_distribution<std::size_t>::param_type other_layer{0, Def::x};

            const auto y_idx = y(g);
            if (y_idx == 0)
                circuit[y_idx][x(g)].connectors.flip(dst(g, first_layer));
            else
                circuit[y_idx][x(g)].connectors.flip(dst(g, other_layer));
        }
    }

private:
    const std::size_t changes_of_functions;
    const std::size_t changes_of_arguments;
    const std::size_t changes_of_connectors;
    const sample_pool<function> function_generator;
};

template <class Def> struct categories_evaluator {
    categories_evaluator(std::size_t categories)
        : _chisqr(categories) {
    }

    double apply(const circuit<Def> &circuit) {
        interpreter<Def> kernel{circuit};

        _out_a.clear();
        _out_b.clear();

        std::transform(_in_a.begin(), _in_a.end(), std::back_inserter(_out_a.begin()), kernel);
        std::transform(_in_b.begin(), _in_b.end(), std::back_inserter(_out_b.begin()), kernel);

        return 1.0 - _chisqr(_out_a, _out_b);
    }

private:
    dataset<Def::in> _in_a;
    dataset<Def::in> _in_b;
    dataset<Def::out> _out_a;
    dataset<Def::out> _out_b;
    two_sample_chisqr _chisqr;
};

template <class Def> struct basic_initializer {
    basic_initializer(const sample_pool<function> &pool)
        : function_generator(pool) {
    }

    template <class Generator> void apply(circuit<Def> &circuit, Generator &g) const {
        // for the first layer...
        for (unsigned i = 0; i != Def::x; ++i) {
            auto node = circuit[0][i];

            node.connectors = (i < Def::in) ? 1u << i : 0;
            node.function = function::XOR;
            node.argument = helper<Def>::generate_argument(g);
        }

        // for the other layers...
        for (unsigned i = 1; i != Def::y; ++i)
            for (auto node : circuit[i]) {
                node.connectors = helper<Def>::generate_connectors(g, Def::x);
                node.function = function_generator(g);
                node.argument = helper<Def>::generate_argument(g);
            }
    }

private:
    const sample_pool<function> function_generator;
};

} // namespace circuit
