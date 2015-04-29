#pragma once

#include "ga_callbacks.h"
#include "circuit/gate/GateCircuit.h"


class gpu_gate :
        public GateCircuit
{
public:
    gpu_gate() = default;
    ~gpu_gate() = default;

public:
    virtual std::string shortDescription() { return "cuda gate circuit emulator"; }

    virtual GAGenome::Evaluator getEvaluator() { return ga_callbacks::evaluator; }
};
