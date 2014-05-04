/* 
 * File:   Repr.cpp
 * Author: ph4r05
 * 
 * Created on April 29, 2014, 3:00 PM
 */

#include "ICircuit.h"
#include "gate/GateCircuit.h"
#include "polynomial/PolynomialCircuit.h"

ICircuit::ICircuit(int type) : m_type(type), io(NULL) {
}

ICircuit::~ICircuit() {
}

int ICircuit::getCircuitType() const {
    return m_type;
}

ICircuit* ICircuit::getCircuit(int circuitType) {
    ICircuit* circuit = NULL;
    switch (circuitType) {
    case CIRCUIT_GATE:
        circuit = new CircuitRepr();
        break;
    case CIRCUIT_POLYNOMIAL:
        circuit = new PolyRepr();
        break;
    default:
        mainLogger.out(LOGGER_ERROR) << "Cannot initialize circuit representation - unknown type (" << circuitType << ")." << endl;
        return NULL;
        break;
    }
    mainLogger.out(LOGGER_INFO) << "Circuit representation successfully initialized. (" << circuit->shortDescription() << ")" << endl;
    return circuit;
}
