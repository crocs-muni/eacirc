/**
 * @file GateCircuit.cpp
 * @author Martin Ukrop, ph4r05
 */

#include "GateCircuit.h"
#include "GateCircuitIO.h"
#include "GateCommonFunctions.h"
#include "GAGateCallbacks.h"
#include "GateInterpreter.h"
#include "GAPopulation.h"
#include "XMLProcessor.h"

#define max(a,b) (((a)>(b))?(a):(b))

GateCircuit::GateCircuit() : ICircuit(CIRCUIT_GATE) { }

GateCircuit::~GateCircuit() {
    if (executionInputLayer != NULL) delete[] executionInputLayer;
    executionInputLayer = NULL;
    if (executionMiddleLayerIn != NULL) delete[] executionMiddleLayerIn;
    executionMiddleLayerIn = NULL;
    if (executionMiddleLayerOut != NULL) delete[] executionMiddleLayerOut;
    executionMiddleLayerOut = NULL;
    if (executionOutputLayer != NULL) delete[] executionOutputLayer;
    executionOutputLayer = NULL;
}

string GateCircuit::shortDescription() {
    return "gate circuit emulator";
}

int GateCircuit::initialize() {
    executionInputLayer = new unsigned char[pGlobals->settings->gateCircuit.sizeInputLayer];
    executionMiddleLayerIn = new unsigned char[pGlobals->settings->gateCircuit.sizeLayer];
    executionMiddleLayerOut = new unsigned char[pGlobals->settings->gateCircuit.sizeLayer];
    executionOutputLayer = new unsigned char[pGlobals->settings->gateCircuit.sizeOutputLayer];
    return STAT_OK;
}

inline GAGenome::Initializer       GateCircuit::getInitializer() { return GAGateCallbacks::initializer; }
inline GAGenome::Evaluator         GateCircuit::getEvaluator()   { return GAGateCallbacks::evaluator;   }
inline GAGenome::Mutator           GateCircuit::getMutator()     { return GAGateCallbacks::mutator;     }
inline GAGenome::Comparator        GateCircuit::getComparator()  { return NULL; }
inline GAGenome::SexualCrossover   GateCircuit::getSexualCrossover()  { return GAGateCallbacks::crossover; }
inline GAGenome::AsexualCrossover  GateCircuit::getAsexualCrossover() { return NULL; }

GAGenome* GateCircuit::createGenome(bool setCallbacks) {
    GA1DArrayGenome<GENOME_ITEM_TYPE> *g = new GA1DArrayGenome<GENOME_ITEM_TYPE>(pGlobals->settings->gateCircuit.genomeSize, getEvaluator());
    if (setCallbacks){
        setGACallbacks(g);
    }

    return g;
}

GAPopulation* GateCircuit::createPopulation() {
    GA1DArrayGenome<GENOME_ITEM_TYPE> g(pGlobals->settings->gateCircuit.genomeSize, getEvaluator());
    setGACallbacks(&g);

    GAPopulation * population = new GAPopulation(g, pGlobals->settings->ga.popupationSize);
    return population;
}

bool GateCircuit::postProcess(GAGenome& original, GAGenome& prunned) {
    if (!pGlobals->settings->outputs.allowPrunning) {
        return false;
    }
    int status = GateInterpreter::pruneCircuit(original, prunned);
    if (status != STAT_OK) {
        mainLogger.out(LOGGER_WARNING) << "Could not post-process genome (" << status << ")." << endl;
        return false;
    }
    return true;
}

int GateCircuit::loadCircuitConfiguration(TiXmlNode* pRoot) {
    // parsing EACIRC/GATE_CIRCUIT
    pGlobals->settings->gateCircuit.numLayers = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/NUM_LAYERS").c_str());
    pGlobals->settings->gateCircuit.sizeLayer = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/SIZE_LAYER").c_str());
    pGlobals->settings->gateCircuit.numConnectors = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/NUM_CONNECTORS").c_str());
    pGlobals->settings->gateCircuit.useMemory = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/USE_MEMORY").c_str()) ? true : false;
    pGlobals->settings->gateCircuit.sizeMemory = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/SIZE_MEMORY").c_str());
    // parsing EACIRC/GATE_CIRCUIT/ALLOWED_FUNCTIONS
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_NOP] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_NOP").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_CONS] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_CONS").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_AND] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_AND").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_NAND] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_NAND").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_OR] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_OR").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_XOR] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_XOR").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_NOR] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_NOR").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_NOT] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_NOT").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_SHIL] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_SHIL").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_SHIR] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_SHIR").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_ROTL] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_ROTL").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_ROTR] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_ROTR").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_EQ] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_EQ").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_LT] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_LT").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_GT] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_GT").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_LEQ] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_LEQ").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_GEQ] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_GEQ").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_BSLC] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_BSLC").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_READ] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_READ").c_str());
    pGlobals->settings->gateCircuit.allowedFunctions[FNC_JVM] = atoi(getXMLElementValue(pRoot,"GATE_CIRCUIT/ALLOWED_FUNCTIONS/FNC_JVM").c_str());

    // update extra info
    if (!pGlobals->settings->gateCircuit.useMemory) {
        pGlobals->settings->gateCircuit.sizeMemory = 0;
    }
    pGlobals->settings->gateCircuit.sizeOutputLayer = pGlobals->settings->main.circuitSizeOutput + pGlobals->settings->gateCircuit.sizeMemory;
    pGlobals->settings->gateCircuit.sizeInputLayer = pGlobals->settings->main.circuitSizeInput + pGlobals->settings->gateCircuit.sizeMemory;
    pGlobals->settings->gateCircuit.genomeWidth = max(pGlobals->settings->gateCircuit.sizeLayer, pGlobals->settings->gateCircuit.sizeOutputLayer);
    // Compute genome size: genomeWidth for number of layers (each layer is twice - function and connector)
    pGlobals->settings->gateCircuit.genomeSize = pGlobals->settings->gateCircuit.numLayers * 2 * pGlobals->settings->gateCircuit.genomeWidth;

    if (pGlobals->settings->gateCircuit.sizeLayer > MAX_LAYER_SIZE || pGlobals->settings->gateCircuit.numConnectors > MAX_LAYER_SIZE
            || pGlobals->settings->gateCircuit.sizeInputLayer > MAX_LAYER_SIZE || pGlobals->settings->gateCircuit.sizeOutputLayer > MAX_LAYER_SIZE) {
        mainLogger.out(LOGGER_ERROR) << "Maximum layer size exceeded (internal size || connectors || total input|| total output)." << endl;
        return STAT_CONFIG_INCORRECT;
    }
    if (pGlobals->settings->gateCircuit.useMemory && pGlobals->settings->gateCircuit.sizeMemory <= 0) {
        mainLogger.out(LOGGER_ERROR) << "Memory enabled but size incorrectly set (negative or zero)." << endl;
        return STAT_CONFIG_INCORRECT;
    }

    return STAT_OK;
}
