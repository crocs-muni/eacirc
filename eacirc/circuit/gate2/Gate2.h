#pragma once

#include "GaGate.h"
#include "circuit/gate/GateCircuit.h"


class Gate2 :
    public GateCircuit
{
public:
    Gate2() : GateCircuit( CIRCUIT_GATE2 ) {}
    ~Gate2() = default;
public:
    std::string shortDescription() override;

    GAGenome::Evaluator getEvaluator() override { return GaGate::evaluator; }

    int loadCircuitConfiguration( TiXmlNode* pRoot );
};