#include "GpuTask.h"

#include "GateInterpreter.h"
#include <cuda_runtime_api.h>


using GenomeItem = GpuTask::GenomeItem;
using CircuitType = GateCircuit<GenomeItem>;
using CircuitNode = typename CircuitType::Node;

constexpr size_t circuitMaxNodes = 700;


__constant__ CircuitNode devCircuitNodes[circuitMaxNodes];


template <class T>
__global__ void kernel(const Byte* ins, Byte* outs, const GateCircuit<T> circuit, const size_t bankSize)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    // get offset to inputs and outputs
    const Byte* in = ins + (id * circuit.inSize);
    Byte* out = outs + (id * circuit.outSize);
    
    // allocate memory for execution layer
    extern __shared__ Byte memory[];
    Byte* layers = memory + ((2 * circuit.inSize + bankSize) * threadIdx.x);

    // execute
    GateInterpreter<T> interpreter(layers, &circuit);
    if (!interpreter.execute(in, out))
        return; //TODO report error
}




void GpuTask::updateCircuit(const CircuitNode* nodes)
{
    const size_t nodeCount = circuit.layerNum * circuit.genomeWidth;

    cudaMemcpyToSymbol(devCircuitNodes, nodes, nodeCount * sizeof(CircuitNode));
    cudaGetSymbolAddress((void**)(&(circuit.data)), devCircuitNodes);
}


void GpuTask::run()
{
    const size_t gridSize = vecCount / blockSize;
    const size_t memSize = blockSize * (2 * circuit.inSize + bankSize);

    kernel<GenomeItem><<< gridSize,  blockSize, memSize >>>( devIns, devOuts, circuit, bankSize );
}
