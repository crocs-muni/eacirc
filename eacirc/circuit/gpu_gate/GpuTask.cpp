#include "GpuTask.h"

#include "GateInterpreter.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <cmath>


GpuTask::GpuTask(const CircuitType& circuit, const size_t vecCount, const size_t blockSize) :
    vecCount(static_cast<size_t>(std::ceil(float(vecCount) / blockSize) * blockSize)),
    bankSize(0),
    blockSize(blockSize),
    devIns(nullptr),
    devOuts(nullptr),
    hostIns(nullptr),
    hostOuts(nullptr),
    circuit(circuit)
{
    cudaSetDevice(0);

    cudaMallocHost(&hostIns, circuit.inSize * vecCount);
    cudaMallocHost(&hostOuts, circuit.outSize * vecCount);

    cudaMalloc(&devIns, circuit.inSize * vecCount);
    cudaMalloc(&devOuts, circuit.outSize * vecCount);

    cudaSharedMemConfig config;
    cudaDeviceGetSharedMemConfig(&config);
    bankSize = (config == cudaSharedMemBankSizeEightByte) ? 8 : 4;
}

GpuTask::~GpuTask()
{
    cudaFree(devIns);
    cudaFree(devOuts);

    cudaFreeHost(hostIns);
    cudaFreeHost(hostOuts);
}


void GpuTask::updateInputs(const Byte** ins, const size_t n)
{
    for ( size_t i = 0; i < n; ++i) {
        std::memcpy(hostIns + (i * circuit.inSize), ins[i], circuit.inSize);
    }
    cudaMemcpy(devIns, hostIns, circuit.inSize * n, cudaMemcpyHostToDevice);
}

Byte* GpuTask::receiveOutputs()
{
    cudaMemcpy(hostOuts, devOuts, circuit.outSize * vecCount, cudaMemcpyDeviceToHost);
    return hostOuts;
}
