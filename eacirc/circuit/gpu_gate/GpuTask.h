#pragma once

#include "Byte.h"
#include "GateCircuit.h"
#include <inttypes.h>


class GpuTask
{
public:
    using GenomeItem = uint32_t;

    using CircuitType = GateCircuit<GenomeItem>;
    using CircuitNode = CircuitType::Node;

public:
    GpuTask(const CircuitType& circuit, const size_t vec_count, const size_t block_size);
    ~GpuTask();

public:
    void updateInputs(const Byte** ins, const size_t n);
    void updateCircuit(const CircuitNode* genome);

    void run();
    Byte* receiveOutputs();

private:
    size_t vecCount;
    size_t bankSize;
    size_t blockSize;

    Byte* devIns;
    Byte* devOuts;

    Byte* hostIns;
    Byte* hostOuts;

    CircuitType circuit;
};
