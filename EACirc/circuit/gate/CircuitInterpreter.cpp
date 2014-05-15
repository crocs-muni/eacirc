#include "CircuitInterpreter.h"

unsigned char* executionInputLayer = NULL;
unsigned char* executionMiddleLayerIn = NULL;
unsigned char* executionMiddleLayerOut = NULL;
unsigned char* executionOutputLayer = NULL;

int CircuitInterpreter::executeCircuit(GA1DArrayGenome<GENOME_ITEM_TYPE>* pGenome, unsigned char* inputs, unsigned char* outputs) {
    // allocate repeatedly used variables
    int offsetConnectors;
    int offsetFunctions;
    GENOME_ITEM_TYPE absoluteConnectors;
    GENOME_ITEM_TYPE function;
    int status;

    // initial memory inputs are zero
    memset(executionOutputLayer, 0, pGlobals->settings->gateCircuit.sizeOutputLayer);

    // compute number of memory cycles
    int numMemoryCycles = !pGlobals->settings->gateCircuit.useMemory ? 1 : pGlobals->settings->testVectors.inputLength / pGlobals->settings->main.circuitSizeInput;

    // execute entire circuit for each memory cycle
    for (int memoryCycle = 0; memoryCycle < numMemoryCycles; memoryCycle++) {
        // prepare initial input layer values (memory + inputs)
        memcpy(executionInputLayer, executionOutputLayer, pGlobals->settings->gateCircuit.sizeMemory);
        memcpy(executionInputLayer + pGlobals->settings->gateCircuit.sizeMemory, inputs, pGlobals->settings->main.circuitSizeInput);

        // execute first layer
        offsetConnectors = 0;
        offsetFunctions = pGlobals->settings->gateCircuit.genomeWidth;
        for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeLayer; slot++) {
            absoluteConnectors = relativeToAbsoluteConnectorMask(pGenome->gene(offsetConnectors + slot), slot, pGlobals->settings->gateCircuit.sizeInputLayer,
                                                                 pGlobals->settings->gateCircuit.sizeInputLayer);
            function = pGenome->gene(offsetFunctions + slot);
            status = executeFunction(function, absoluteConnectors, executionInputLayer,
                                     (executionMiddleLayerOut[slot]));
            if (status != STAT_OK) return status;
        }
        // copy results to new inputs
        memcpy(executionMiddleLayerIn, executionMiddleLayerOut, pGlobals->settings->gateCircuit.sizeLayer);

        // execute inside layers
        for (int layer = 1; layer < pGlobals->settings->gateCircuit.numLayers - 1; layer++) {
            offsetConnectors = 2 * layer * pGlobals->settings->gateCircuit.genomeWidth;
            offsetFunctions = offsetConnectors + pGlobals->settings->gateCircuit.genomeWidth;
            for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeLayer; slot++) {
                absoluteConnectors = relativeToAbsoluteConnectorMask(pGenome->gene(offsetConnectors + slot), slot, pGlobals->settings->gateCircuit.sizeLayer,
                                                                     pGlobals->settings->gateCircuit.numConnectors);
                function = pGenome->gene(offsetFunctions + slot);
                status = executeFunction(function, absoluteConnectors, executionMiddleLayerIn,
                                         (executionMiddleLayerOut[slot]));
                if (status != STAT_OK) return status;
            }
            // copy results to new inputs
            memcpy(executionMiddleLayerIn, executionMiddleLayerOut, pGlobals->settings->gateCircuit.sizeLayer);
        }

        // execute last layer
        offsetConnectors = 2 * (pGlobals->settings->gateCircuit.numLayers - 1) * pGlobals->settings->gateCircuit.genomeWidth;
        offsetFunctions = offsetConnectors + pGlobals->settings->gateCircuit.genomeWidth;
        for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeOutputLayer; slot++) {
            absoluteConnectors = relativeToAbsoluteConnectorMask(pGenome->gene(offsetConnectors + slot), slot % pGlobals->settings->gateCircuit.sizeLayer,
                                                                 pGlobals->settings->gateCircuit.sizeLayer, pGlobals->settings->gateCircuit.sizeLayer);
            function = pGenome->gene(offsetFunctions + slot);
            status = executeFunction(function, absoluteConnectors, executionMiddleLayerIn,
                                     (executionOutputLayer[slot]));
            if (status != STAT_OK) return status;
        }

    }

    // copy final outputs
    memcpy(outputs, executionOutputLayer + pGlobals->settings->gateCircuit.sizeMemory, pGlobals->settings->main.circuitSizeOutput);
    return STAT_OK;
}

int CircuitInterpreter::pruneCircuit(GAGenome &originalGenome, GAGenome &prunnedGenome) {
    return STAT_NOT_IMPLEMENTED_YET;
}

int CircuitInterpreter::executeFunction(GENOME_ITEM_TYPE node, GENOME_ITEM_TYPE absoluteConnectors, unsigned char* layerInputValues, unsigned char& result) {
    // temporary variables
    int connection = 0;
    int connection2 = 0;
    unsigned char function = nodeGetFunction(node);
    unsigned char argument1 = nodeGetArgument(node,1);
    // start result with neutral value
    result = getNeutralValue(function);

    switch (function) {
    case FNC_NOP:
        if (connectorsDiscartFirst(absoluteConnectors,connection)) { result = layerInputValues[connection]; }
        break;
    case FNC_CONS:
        result = argument1;
        break;
    case FNC_AND:
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result &= layerInputValues[connection]; }
        break;
    case FNC_NAND:
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result &= layerInputValues[connection]; }
        result = ~result;
        break;
    case FNC_OR:
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result |= layerInputValues[connection]; }
        break;
    case FNC_XOR:
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result ^= layerInputValues[connection]; }
        break;
    case FNC_NOR:
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result |= layerInputValues[connection]; }
        result = ~result;
        break;
    case FNC_NOT:
        if (connectorsDiscartFirst(absoluteConnectors,connection)) { result = ~(layerInputValues[connection]); }
        break;
    case FNC_SHIL:
        if (connectorsDiscartFirst(absoluteConnectors,connection)) { result = layerInputValues[connection] << (argument1 % BITS_IN_UCHAR); }
        break;
    case FNC_SHIR:
        if (connectorsDiscartFirst(absoluteConnectors,connection)) { result = layerInputValues[connection] >> (argument1 % BITS_IN_UCHAR); }
        break;
    case FNC_ROTL:
        if (connectorsDiscartFirst(absoluteConnectors,connection)) {
            result = (layerInputValues[connection] << (argument1 % BITS_IN_UCHAR))
                      | (layerInputValues[connection] >> (BITS_IN_UCHAR - argument1 % BITS_IN_UCHAR));
        }
        break;
    case FNC_ROTR:
        if (connectorsDiscartFirst(absoluteConnectors,connection)) {
            result = (layerInputValues[connection] >> (argument1 % BITS_IN_UCHAR))
                      | (layerInputValues[connection] << (BITS_IN_UCHAR - argument1 % BITS_IN_UCHAR));
        }
        break;
    case FNC_EQ:
        if (connectorsDiscartFirst(absoluteConnectors,connection) && connectorsDiscartFirst(absoluteConnectors,connection2)) {
            if (layerInputValues[connection] == layerInputValues[connection2]) { result = UCHAR_MAX; }
        }
        break;
    case FNC_LT:
        if (connectorsDiscartFirst(absoluteConnectors,connection) && connectorsDiscartFirst(absoluteConnectors,connection2)) {
            if (layerInputValues[connection] < layerInputValues[connection2]) { result = UCHAR_MAX; }
        }
        break;
    case FNC_GT:
        if (connectorsDiscartFirst(absoluteConnectors,connection) && connectorsDiscartFirst(absoluteConnectors,connection2)) {
            if (layerInputValues[connection] > layerInputValues[connection2]) { result = UCHAR_MAX; }
        }
        break;
    case FNC_LEQ:
        if (connectorsDiscartFirst(absoluteConnectors,connection) && connectorsDiscartFirst(absoluteConnectors,connection2)) {
            if (layerInputValues[connection] <= layerInputValues[connection2]) { result = UCHAR_MAX; }
        }
        break;
    case FNC_GEQ:
        if (connectorsDiscartFirst(absoluteConnectors,connection) && connectorsDiscartFirst(absoluteConnectors,connection2)) {
            if (layerInputValues[connection] >= layerInputValues[connection2]) { result = UCHAR_MAX; }
        }
        break;
    case FNC_BSLC:
        if (connectorsDiscartFirst(absoluteConnectors,connection)) { result = layerInputValues[connection] & argument1; }
        break;
    case FNC_READ:
        result = executionInputLayer[argument1 % pGlobals->settings->gateCircuit.sizeInputLayer];
        break;
    case FNC_EXT:
        return executeExternalFunction(node, absoluteConnectors, layerInputValues, result);

        // unknown function constant
    default:
        mainLogger.out(LOGGER_ERROR) << "Unknown circuit function (" << nodeGetFunction(function) << ")." << endl;
        return STAT_CIRCUIT_INCONSISTENT;
    }
    return STAT_OK;
}

int CircuitInterpreter::executeExternalFunction(GENOME_ITEM_TYPE node, GENOME_ITEM_TYPE absoluteConnectors, unsigned char* layerInputValues, unsigned char &result) {
    return STAT_NOT_IMPLEMENTED_YET;
}
