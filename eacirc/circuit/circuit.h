#pragma once

#include "connectors.h"
#include "functions.h"
#include <core/vec.h>

#include <iostream>
#include <fstream>
#include <string>

namespace circuit {

    template <unsigned DimX, unsigned DimY, unsigned Out> struct circuit {
        static constexpr unsigned x = DimX;
        static constexpr unsigned y = DimY;

        using output = vec<Out>;
        using connectors_type = connectors<32>;

        struct node {
            fn function{fn::NOP};
            std::uint8_t argument{0u};
            connectors_type connectors{0u};
            bool used{true};
        };

        using layer = std::array<node, x>;
        using layers = std::array<layer, y>;

        using iterator = typename layers::iterator;
        using const_iterator = typename layers::const_iterator;

        circuit(unsigned input)
            : _input(input), _input_used(input, true) {}

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

        void dump_to_graph(const std::string &filename) const {
            std::ofstream of(filename);

            // graph header
            of << "graph EACircuit {" << std::endl << "rankdir=BT;" << std::endl << "ranksep=0.75;" << std::endl << "ordering=out;" << std::endl;
            of << "splines=polyline;" << std::endl << "node [style=filled, color=lightblue2];" << std::endl << std::endl;

            // node specification
            // input nodes
            of << "{ rank=same;" << std::endl;
            for (std::size_t slot = 0; slot < _input; ++slot) {
                if (_input_used[slot]) {
                    of << "node [color=chartreuse3];" << std::endl;
                } else {
                    of << "node [color=chartreuse1];" << std::endl;
                }
                of << "\"-1_" << slot << "\"[label=\"" << "IN" << "\\n" << slot << "\"];" << std::endl;
            }
            of << "}" << std::endl;

            // inside nodes
            std::size_t layer_num = 0;
            for (auto l : _layers) {
                of << "{ rank=same;" << std::endl;

                // last layer
                if (layer_num + 1 == _layers.size()) {
                    of << "node [color=brown2];" << std::endl;
                    of << "\"" << layer_num << "_0\"[label=\"";
                    of << to_string(l[0].function) << "\\n" << int(l[0].argument) << "\"];" << std::endl;
                    of << "}" << std::endl;
                    break;
                }

                std::size_t slot_num = 0;
                for (auto n : l) {
                    if (n.used) {
                        of << "node [color=lightblue3];" << std::endl;
                    } else {
                        of << "node [color=lightblue1];" << std::endl;
                    }
                    of << "\"" << layer_num << "_" << slot_num << "\"[label=\"";
                    of << to_string(n.function) << "\\n" << int(n.argument) << "\"];" << std::endl;
                    ++slot_num;
                }

                of << "}" << std::endl;
                ++layer_num;
            }

            /// connectors

            // invisible connectors (to preserve order in rows)
            of << "edge[style=invis];" << std::endl;

            // input nodes
            of << "\"-1_0\"";
            for (std::size_t slot = 1; slot < _input; ++slot) {
                of << " -- \"-1_" << slot << "\"";
            }
            of << ";" << std::endl;

            // inside nodes
            layer_num = 0;
            for (auto l : _layers) {
                of << "\"" << layer_num << "_0\"";

                // last layer
                if (layer_num + 1 == _layers.size()) {
                    of << " -- \"" << layer_num << "_0\";" << std::endl;
                    break;
                }

                std::size_t slot_num = 0;
                for (auto _ : l) {
                    if (slot_num != 0) // first node was written manually above
                        of << " -- \"" << layer_num << "_" << slot_num << "\"";
                    ++slot_num;
                }

                of << ";" << std::endl;
                ++layer_num;
            }

            // normal connectors
            of << "edge[style=solid];" << std::endl;

            layer_num = 0;
            for (auto l : _layers) {

                std::size_t slot_num = 0;
                for (auto n : l) {

                    for (auto it = n.connectors.iterator(); it.has_next(); it.next()) {
                        of << "\"" << layer_num << "_" << slot_num << "\" -- \"" << (int(layer_num) - 1) << "_" << it << "\";" << std::endl;
                    }

                    // last layer
                    if (layer_num + 1 == _layers.size())
                        break;

                    ++slot_num;
                }

                ++layer_num;
            }

            // footer & close
            of << "}";
            of.close();
        }

        void prune() {
            // BFS - set all nodes unvisited
            for (auto &l : _layers)
                for (auto &n : l)
                    n.used = false;
            for (auto &&u : _input_used) // C++ specialization of std::vector<bool> - proxy iterator
                u = false;

            // the single output node is used
            _layers[_layers.size() - 1][0].used = true;


            // l_i is by 1 more, than current layer number, to hold even for input layer
            for (std::size_t l_i = _layers.size(); l_i != 0; --l_i) {
                for (auto &n : _layers[l_i-1]) {
                    if (!n.used) {
                        n.connectors = 0u;
                        continue;
                    }

                    std::size_t arity = fn_arity(n.function);
                    auto it = n.connectors.iterator();

                    while (arity != 0 && it.has_next()) {
                        if (l_i == 1) {
                            _input_used[it] = true;
                        } else {
                            _layers[l_i - 2][it].used = true;
                        }
                        --arity;
                        it.next();
                    }

                    while (it.has_next()) {
                        n.connectors.clear(it);
                        it.next();
                    }
                }
            }
        }

    private:
        layers _layers;
        unsigned _input;
        std::vector<bool> _input_used;
    };

} // namespace circuit
