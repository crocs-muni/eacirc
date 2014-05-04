/**
 * @file ICircuit.cpp
 * @author Martin Ukrop, ph4r05
 */

#include "ICircuit.h"
#include "gate/GateCircuit.h"
#include "gate/GateCircuitIO.h"
#include "polynomial/PolynomialCircuit.h"
#include "polynomial/PolynomialCircuitIO.h"

ICircuit::ICircuit(int type) : m_type(type), ioCallbackObject(NULL) {
}

ICircuit::~ICircuit() {
    delete ioCallbackObject;
    ioCallbackObject = NULL;
}

ICircuitIO * ICircuit::io() {
    return ioCallbackObject;
}

int ICircuit::getCircuitType() const {
    return m_type;
}

void ICircuit::setGACallbacks(GAGenome * g) {
    g->initializer(getInitializer());
    g->evaluator(getEvaluator());
    g->mutator(getMutator());
    g->comparator(getComparator());
    if (getSexualCrossover() != NULL) {
        g->crossover(getSexualCrossover());
    } else {
        g->crossover(getAsexualCrossover());
    }
}

bool ICircuit::postProcess(GAGenome &original, GAGenome &processed) {
    return false;
}

int ICircuit::loadCircuitConfiguration(TiXmlNode* pRoot) {
    return STAT_OK;
}

ICircuit* ICircuit::getCircuit(int circuitType) {
    ICircuit* circuit = NULL;
    switch (circuitType) {
    case CIRCUIT_GATE:
        circuit = new GateCircuit();
        circuit->ioCallbackObject = new CircuitIO();
        break;
    case CIRCUIT_POLYNOMIAL:
        circuit = new PolynomialCircuit();
        circuit->ioCallbackObject = new PolyIO();
        break;
    default:
        mainLogger.out(LOGGER_ERROR) << "Cannot initialize circuit representation - unknown type (" << circuitType << ")." << endl;
        return NULL;
        break;
    }
    mainLogger.out(LOGGER_INFO) << "Circuit representation successfully initialized. (" << circuit->shortDescription() << ")" << endl;
    return circuit;
}
