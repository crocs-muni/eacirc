#pragma once

#include "../statistics.h"
#include "circuit.h"
#include "interpreter.h"
#include <algorithm>
#include <core/dataset.h>
#include <core/json.h>
#include <random>

namespace circuit {

    template <typename Generator> static std::uint8_t generate_argument(Generator& g) {
        std::uniform_int_distribution<std::uint8_t> dst;
        return dst(g);
    }

    template <typename Connectors, typename Generator>
    static Connectors generate_connetors(Generator& g, unsigned size) {
        std::uniform_int_distribution<typename Connectors::value_type> dst{0, (1u << size) - 1};
        return Connectors{dst(g)};
    }

    struct basic_mutator {
        basic_mutator(json const& config, fn_set function_set)
            : _changes_of_functions(config.at("changes-of-functions"))
            , _changes_of_arguments(config.at("changes-of-arguments"))
            , _changes_of_connectors(config.at("changes-of-connectors"))
            , _function_set(std::move(function_set)) {}

        template <typename Circuit, typename Generator> void apply(Circuit& circuit, Generator& g) {
            std::uniform_int_distribution<std::size_t> x{0, Circuit::x - 1};
            std::uniform_int_distribution<std::size_t> y{0, Circuit::y - 1};

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
                using uniform_distribution = std::uniform_int_distribution<unsigned>;

                uniform_distribution dst;
                uniform_distribution::param_type first_layer{0, circuit.input() - 1};
                uniform_distribution::param_type other_layer{0, Circuit::x - 1};

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

    template <typename Circuit, typename Generator> void reduce(Circuit& circuit, Generator& g, std::size_t count) {
        std::uniform_int_distribution<std::size_t> x{0, Circuit::x - 1};
        std::uniform_int_distribution<std::size_t> y{0, Circuit::y - 1};

        // mutate connectors
        for (std::size_t i = 0; i != count; ++i) {
            using uniform_distribution = std::uniform_int_distribution<unsigned>;

            uniform_distribution dst;
            uniform_distribution::param_type first_layer{0, circuit.input() - 1};
            uniform_distribution::param_type other_layer{0, Circuit::x - 1};

            const auto y_idx = y(g);
            if (y_idx == 0)
                circuit[y_idx][x(g)].connectors.clear(dst(g, first_layer));
            else
                circuit[y_idx][x(g)].connectors.clear(dst(g, other_layer));
        }
    }

    template <typename Circuit, typename Generator> void extend(Circuit& circuit, Generator& g, std::size_t count) {
        std::uniform_int_distribution<std::size_t> x{0, Circuit::x - 1};
        std::uniform_int_distribution<std::size_t> y{0, Circuit::y - 1};

        // mutate connectors
        for (std::size_t i = 0; i != count; ++i) {
            using uniform_distribution = std::uniform_int_distribution<unsigned>;

            uniform_distribution dst;
            uniform_distribution::param_type first_layer{0, circuit.input() - 1};
            uniform_distribution::param_type other_layer{0, Circuit::x - 1};

            const auto y_idx = y(g);
            if (y_idx == 0)
                circuit[y_idx][x(g)].connectors.set(dst(g, first_layer));
            else
                circuit[y_idx][x(g)].connectors.set(dst(g, other_layer));
        }
    }

    struct basic_initializer {
        basic_initializer(json const&, fn_set function_set)
            : _function_set(std::move(function_set)) {}

        template <typename Circuit, typename Generator> void apply(Circuit& circuit, Generator& g) {
            // for the first layer...
            for (unsigned i = 0; i != Circuit::x; ++i) {
                auto& node = circuit[0][i];

                node.connectors = (i < circuit.input()) ? (1u << i) : 0u;
                node.function = fn::XOR;
                node.argument = generate_argument(g);
            }

            // for the other layers...
            for (unsigned i = 1; i != Circuit::y; ++i)
                for (auto& node : circuit[i]) {
                    using connectors = typename Circuit::connectors_type;
                    node.connectors = generate_connetors<connectors>(g, Circuit::x);
                    node.function = _function_set.choose(g);
                    node.argument = generate_argument(g);
                }
        }

    private:
        const fn_set _function_set;
    };
} // namespace circuit
