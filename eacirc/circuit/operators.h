#pragma once

#include "circuit.h"
#include "settings.h"
#include <random>

namespace circuit {
    struct Initializer {
        template <class T> void operator()(T& circuit);
    };

    class ArgumentGenerator {
        std::uniform_int_distribution<ui8> _dist;

    public:
        template <class Generator> void operator()(Generator& g)
        {
            return _dist(g);
        }
    };

    class FunctionGenerator {
        using Type = typename std::underlying_type<Function>::type;

        Settings const& _settings;
        std::uniform_int_distribution<Type> _dist;

    public:
        FunctionGenerator(Settings const& settings)
                : _settings(settings), _dist(0, to_underlying(Function::_Size))
        {
        }

        template <class Generator> Function operator()(Generator& g)
        {
            Type fn;
            do {
                fn = _dist(g);
            } while (!_settings.function_set[fn]);
            return static_cast<Function>(fn);
        }
    };

    class Mutator {
        Settings const& _settings;

    public:
        Mutator(Settings const& settings) : _settings(settings) {}

        template <class Circuit, class Generator>
        void operator()(Circuit& circuit, Generator& g)
        {
            const auto x = Circuit::x;
            const auto y = Circuit::y;

            unsigned cns_count = (_settings.isize * x) + x * x * (y - 1);
            std::uniform_int_distribution<unsigned> cn_dist(0, cns_count - 1);
            std::uniform_int_distribution<unsigned> fn_dist{
                    0, circuit.node_count() - 1};
            std::uniform_int_distribution<unsigned> arg_dist{
                    0, circuit.node_count() - 1};

            // mutate connectors
            for (...) {
                // FIXME num of connectors is different than you think
                auto i = cn_dist(g);
                circuit.node(i / x).connector.flip(i % x);
            }

            // mutate functions
            for (...) {
                auto i = fn_dist(g);
                circuit.node(i).function = _fn_mutator.generate_fn(g);
            }

            // mutate argument
            for (...) {
                auto i = arg_dist(g);
            }
        }
    };
}
