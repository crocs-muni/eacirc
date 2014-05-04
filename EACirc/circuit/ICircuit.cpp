/* 
 * File:   Repr.cpp
 * Author: ph4r05
 * 
 * Created on April 29, 2014, 3:00 PM
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
