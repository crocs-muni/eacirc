#pragma once

#include "GaCallbacks.h"
#include "circuit/gate/GateCircuit.h"


class GpuGate :
        public GateCircuit
{
public:
    GpuGate() { check_settings(); }
    ~GpuGate() = default;

public:
    virtual std::string shortDescription() { return "cuda gate circuit emulator"; }

    virtual GAGenome::Evaluator getEvaluator() { return GaCallbacks::evaluator; }
private:
    /**
     * Fail if incompatible settings detected.
     */
    void check_settings() const;
};
