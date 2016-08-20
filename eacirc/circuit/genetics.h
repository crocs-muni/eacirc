#pragma once

#include "../dataset.h"
#include "../statistics.h"
#include "circuit.h"
#include "interpreter.h"
#include <algorithm>
#include <core/stream.h>
#include <iterator>
#include <random>

namespace circuit {

template <typename Generator> static std::uint8_t generate_argument(Generator& g) {
    std::uniform_int_distribution<std::uint8_t> dst;
    return dst(g);
}

template <typename Connectors, std::size_t Size, typename Generator>
static Connectors generate_connetors(Generator& g) {
    std::uniform_int_distribution<typename Connectors::value_type> dst{0, (1u << Size) - 1};
    return Connectors{dst(g)};
}

template <typename Circuit> struct basic_mutator {
    basic_mutator(fn_set const& function_set)
        : _changes_of_functions(2)
        , _changes_of_arguments(2)
        , _changes_of_connectors(3)
        , _function_set{function_set} {
    }

    template <typename Generator> void apply(Circuit& circuit, Generator& g) {
        std::uniform_int_distribution<std::size_t> x{0, Circuit::x};
        std::uniform_int_distribution<std::size_t> y{0, Circuit::y};

        // mutate functions
        for (size_t i = 0; i != _changes_of_functions; ++i) {
            circuit[y(g)][x(g)].function = _function_set.choose(g);
        }

        // mutate arguments
        for (size_t i = 0; i != _changes_of_arguments; ++i) {
            circuit[y(g)][x(g)].argument = generate_argument(g);
        }

        // mutate connectors
        for (size_t i = 0; i != _changes_of_connectors; ++i) {
            std::uniform_int_distribution<std::size_t> dst;
            std::uniform_int_distribution<std::size_t>::param_type first_layer{0, Circuit::in};
            std::uniform_int_distribution<std::size_t>::param_type other_layer{0, Circuit::x};

            const auto y_idx = y(g);
            if (y_idx == 0)
                circuit[y_idx][x(g)].connectors.flip(dst(g, first_layer));
            else
                circuit[y_idx][x(g)].connectors.flip(dst(g, other_layer));
        }
    }

private:
    const std::size_t _changes_of_functions;
    const std::size_t _changes_of_arguments;
    const std::size_t _changes_of_connectors;
    const fn_set _function_set;
};

template <typename Circuit> struct basic_initializer {
    basic_initializer(fn_set const& function_set)
        : _function_set(function_set) {
    }

    template <typename Generator> void apply(Circuit& circuit, Generator& g) {
        // for the first layer...
        for (unsigned i = 0; i != Circuit::x; ++i) {
            auto node = circuit[0][i];

            node.connectors = (i < Circuit::in) ? (1u << i) : 0u;
            node.function = fn::XOR;
            node.argument = generate_argument(g);
        }

        // for the other layers...
        for (unsigned i = 1; i != Circuit::y; ++i)
            for (auto node : circuit[i]) {
                node.connectors = generate_connetors<typename Circuit::connectors, Circuit::x>(g);
                node.function = _function_set.choose(g);
                node.argument = generate_argument(g);
            }
    }

private:
    const fn_set _function_set;
};

template <typename Circuit> struct categories_evaluator {
    categories_evaluator(std::size_t categories)
        : _chisqr(categories) {
    }

    double apply(Circuit const& circuit) {
        interpreter<Circuit> kernel{circuit};

        _out_a.clear();
        _out_b.clear();

        std::transform(_in_a.begin(), _in_a.end(), std::back_inserter(_out_a), kernel);
        std::transform(_in_b.begin(), _in_b.end(), std::back_inserter(_out_b), kernel);

        return 1.0 - _chisqr(_out_a, _out_b);
    }

    void replace_datasets(core::stream& stream_a, core::stream& stream_b) {
        stream_a.read(_in_a.data(), _in_a.size());
        stream_b.read(_in_b.data(), _in_b.size());
    }

private:
    dataset<Circuit::in> _in_a;
    dataset<Circuit::in> _in_b;
    dataset<Circuit::out> _out_a;
    dataset<Circuit::out> _out_b;
    two_sample_chisqr _chisqr;
};

} // namespace circuit
