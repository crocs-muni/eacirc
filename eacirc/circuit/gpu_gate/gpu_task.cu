#include "gpu_task.h"

#include "gate_interpreter.h"
#include <cuda_runtime_api.h>



using genome_item_t = gpu_task::genome_item_t;

using circuit_type = gate_circuit<genome_item_t>;
using circuit_node = typename circuit_type::node;


constexpr size_t circuit_max_nodes = 700;

#include <cstdio>


__constant__ circuit_node dev_circuit_nodes[circuit_max_nodes];



template <class T>
__global__ void kernel(const byte* ins, byte* outs, const gate_circuit<T> circuit, const size_t bank_size)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    const byte* in = ins + (id * circuit.in_size);
    byte* out = outs + (id * circuit.out_size);


    extern __shared__ byte memory[];

    byte* layers = memory + ((2 * circuit.in_size + bank_size) * threadIdx.x);

    gate_interpreter<T> interpreter(layers, &circuit);
    if (!interpreter.execute(in, out))
        return; //TODO report error
}




void gpu_task::update_circuit(const circuit_node* nodes)
{
    const size_t node_count = _circuit.layer_num * _circuit.genome_width;

    cudaMemcpyToSymbol(dev_circuit_nodes, nodes, node_count * sizeof(circuit_node));

    cudaGetSymbolAddress((void**)(&(_circuit.data)), dev_circuit_nodes);
}


void gpu_task::run()
{
    const size_t grid_size = _vec_count / _block_size;
    const size_t mem_size = _block_size * (2 * _circuit.in_size + _bank_size);

    kernel<genome_item_t><<< grid_size,  _block_size, mem_size >>>( _dev_ins, _dev_outs, _circuit, _bank_size );
}
