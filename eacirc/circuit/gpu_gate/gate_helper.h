#pragma once

#include "gate_circuit.h"
#include "EACglobals.h"
#include "circuit/gate/GateCommonFunctions.h"
#include <GA1DArrayGenome.h>
#include <cuda_runtime_api.h>


template <typename S, typename D>
class gate_helper
{
public:
    using ga_circuit = GA1DArrayGenome<S>;

    using circuit_type = gate_circuit<D>;
    using circuit_node = typename circuit_type::node;
public:
    gate_helper() :
        _host_nodes(nullptr)
    {
        const size_t layer_num = pGlobals->settings->gateCircuit.numLayers;
        const size_t genome_width = pGlobals->settings->gateCircuit.genomeWidth;

        cudaMallocHost(&_host_nodes, layer_num * genome_width * sizeof(circuit_node));
    }

    ~gate_helper()
    {
        cudaFreeHost(_host_nodes);
    }

public:
    circuit_node* get_nodes() const { return _host_nodes; }


    void transform(const GAGenome* src)
    {
        const ga_circuit* src_circuit = dynamic_cast<const ga_circuit*>(src);

        const size_t out_size = pGlobals->settings->testVectors.outputLength;
        const size_t layer_size = pGlobals->settings->gateCircuit.sizeLayer;
        const size_t layer_num = pGlobals->settings->gateCircuit.numLayers;
        const size_t genome_width = pGlobals->settings->gateCircuit.genomeWidth;
        const size_t size_layer = pGlobals->settings->gateCircuit.sizeLayer;
        const size_t size_input_layer = pGlobals->settings->gateCircuit.sizeInputLayer;
        const size_t num_connectors = pGlobals->settings->gateCircuit.numConnectors;

        for (size_t layer = 0; layer < layer_num; layer++) {
            size_t offset_conns = 2 * layer * genome_width;
            size_t offset_funcs = offset_conns + genome_width;

            size_t num_slots = (layer != layer_num - 1) ? layer_size : out_size;

            for (size_t i = 0; i < num_slots; i++) {
                D func_mask = src_circuit->gene(offset_funcs + i);
                D conn_mask = src_circuit->gene(offset_conns + i);

                if (layer == 0) {
                    conn_mask = relativeToAbsoluteConnectorMask(conn_mask, i, size_input_layer, size_input_layer);
                } else if (layer == layer_num - 1) {
                    conn_mask = relativeToAbsoluteConnectorMask(conn_mask, i % size_layer, size_layer, size_layer);
                } else {
                    conn_mask = relativeToAbsoluteConnectorMask(conn_mask, i, size_layer, num_connectors);
                }

                _host_nodes[(layer * genome_width) + i].set_func(func_mask);
                _host_nodes[(layer * genome_width) + i].set_conn(conn_mask);
            }
        }
    }

private:
    circuit_node* _host_nodes;
};
