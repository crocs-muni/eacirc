#pragma once

#include "byte.h"
#include "gate_circuit.h"
#include <inttypes.h>


class gpu_task
{
public:
    using genome_item_t = uint32_t;

    using circuit_type = gate_circuit<genome_item_t>;
    using circuit_node = circuit_type::node;

public:
    gpu_task(const circuit_type& circuit, const size_t vec_count, const size_t block_size);
    ~gpu_task();

public:
    void update_inputs(const byte** ins, const size_t n);
    void update_circuit(const circuit_node* genome);

    void run();

    byte* receive_outputs();

protected:
    void deploy_circuit(const circuit_node* nodes);

private:
    size_t _vec_count;
    size_t _bank_size;
    size_t _block_size;

    byte* _dev_ins;
    byte* _dev_outs;

    byte* _host_ins;
    byte* _host_outs;

    circuit_type _circuit;
};
