#pragma once

#include "GaCallbacks.h"
#include "circuit/gate/GateCircuit.h"


class GpuGate :
        public GateCircuit
{
public:
    GpuGate() = default;
    ~GpuGate() = default;

public:
    virtual std::string shortDescription() { return "cuda gate circuit emulator"; }

    virtual GAGenome::Evaluator getEvaluator() { return GaCallbacks::evaluator; }
};
