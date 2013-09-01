#include "CircuitInterpreter.h"

int CircuitInterpreter::executeCircuit(GA1DArrayGenome<GENOME_ITEM_TYPE>* pGenome, unsigned char* inputs, unsigned char* outputs) {
    // allocate repeatedly used variables
    int offsetConnectors;
    int offsetFunctions;
    GENOME_ITEM_TYPE absoluteConnectors;
    GENOME_ITEM_TYPE function;
    int status;

    // initial memory inputs are zero
    memset(pGlobals->testVectors.executionOutputLayer, 0, pGlobals->settings->circuit.sizeOutputLayer);

    // compute number of memory cycles
    int numMemoryCycles = !pGlobals->settings->circuit.useMemory ? 1 : pGlobals->settings->testVectors.inputLength / pGlobals->settings->circuit.sizeInput;

    // execute entire circuit for each memory cycle
    for (int memoryCycle = 0; memoryCycle < numMemoryCycles; memoryCycle++) {
        // prepare initial input layer values (memory + inputs)
        memcpy(pGlobals->testVectors.executionInputLayer, pGlobals->testVectors.executionOutputLayer, pGlobals->settings->circuit.sizeMemory);
        memcpy(pGlobals->testVectors.executionInputLayer + pGlobals->settings->circuit.sizeMemory, inputs, pGlobals->settings->circuit.sizeInput);

        // execute first layer
        offsetConnectors = 0;
        offsetFunctions = pGlobals->settings->circuit.genomeWidth;
        for (int slot = 0; slot < pGlobals->settings->circuit.sizeLayer; slot++) {
            absoluteConnectors = relativeToAbsoluteConnectorMask(pGenome->gene(offsetConnectors + slot), slot, pGlobals->settings->circuit.sizeInputLayer,
                                                                 pGlobals->settings->circuit.sizeInputLayer);
            function = pGenome->gene(offsetFunctions + slot);
            status = executeFunction(function, absoluteConnectors, pGlobals->testVectors.executionInputLayer,
                                     (pGlobals->testVectors.executionMiddleLayerOut[slot]));
            if (status != STAT_OK) return status;
        }
        // copy results to new inputs
        memcpy(pGlobals->testVectors.executionMiddleLayerIn, pGlobals->testVectors.executionMiddleLayerOut, pGlobals->settings->circuit.sizeLayer);

        // execute inside layers
        for (int layer = 1; layer < pGlobals->settings->circuit.numLayers - 1; layer++) {
            offsetConnectors = 2 * layer * pGlobals->settings->circuit.genomeWidth;
            offsetFunctions = offsetConnectors + pGlobals->settings->circuit.genomeWidth;
            for (int slot = 0; slot < pGlobals->settings->circuit.sizeLayer; slot++) {
                absoluteConnectors = relativeToAbsoluteConnectorMask(pGenome->gene(offsetConnectors + slot), slot, pGlobals->settings->circuit.sizeLayer,
                                                                     pGlobals->settings->circuit.numConnectors);
                function = pGenome->gene(offsetFunctions + slot);
                status = executeFunction(function, absoluteConnectors, pGlobals->testVectors.executionMiddleLayerIn,
                                         (pGlobals->testVectors.executionMiddleLayerOut[slot]));
                if (status != STAT_OK) return status;
            }
            // copy results to new inputs
            memcpy(pGlobals->testVectors.executionMiddleLayerIn, pGlobals->testVectors.executionMiddleLayerOut, pGlobals->settings->circuit.sizeLayer);
        }

        // execute last layer
        offsetConnectors = 2 * (pGlobals->settings->circuit.numLayers - 1) * pGlobals->settings->circuit.genomeWidth;
        offsetFunctions = offsetConnectors + pGlobals->settings->circuit.genomeWidth;
        for (int slot = 0; slot < pGlobals->settings->circuit.sizeOutputLayer; slot++) {
            absoluteConnectors = relativeToAbsoluteConnectorMask(pGenome->gene(offsetConnectors + slot), slot % pGlobals->settings->circuit.sizeLayer,
                                                                 pGlobals->settings->circuit.sizeLayer, pGlobals->settings->circuit.sizeLayer);
            function = pGenome->gene(offsetFunctions + slot);
            status = executeFunction(function, absoluteConnectors, pGlobals->testVectors.executionMiddleLayerIn,
                                     (pGlobals->testVectors.executionOutputLayer[slot]));
            if (status != STAT_OK) return status;
        }

    }

    // copy final outputs
    memcpy(outputs, pGlobals->testVectors.executionOutputLayer + pGlobals->settings->circuit.sizeMemory, pGlobals->settings->circuit.sizeOutput);
    return STAT_OK;
}

int CircuitInterpreter::pruneCircuit(GAGenome &originalGenome, GAGenome &prunnedGenome) {
    return STAT_NOT_IMPLEMENTED_YET;
}

int CircuitInterpreter::executeFunction(GENOME_ITEM_TYPE node, GENOME_ITEM_TYPE absoluteConnectors, unsigned char* layerInputValues, unsigned char& result) {
    // temporary variables
    int connection = 0;
    unsigned char function = nodeGetFunction(node);
    unsigned char argument1 = nodeGetArgument(node,1);
    // start result with neutral value
    result = getNeutralValue(function);

    switch (function) {
    case FNC_NOP:
        if (connectorsDiscartFirst(absoluteConnectors,connection)) { result = layerInputValues[connection]; }
        break;
    case FNC_OR:
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result |= layerInputValues[connection]; }
        break;
    case FNC_AND:
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result &= layerInputValues[connection]; }
        break;
    case FNC_CONST:
        result = argument1;
        break;
    case FNC_XOR:
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result ^= layerInputValues[connection]; }
        break;
    case FNC_NOR:
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result |= layerInputValues[connection]; }
        result = ~result;
        break;
    case FNC_NAND:
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result &= layerInputValues[connection]; }
        result = ~result;
        break;
    case FNC_ROTL:
        if (connectorsDiscartFirst(absoluteConnectors,connection)) { result = layerInputValues[connection] << (argument1 % BITS_IN_UCHAR); }
        break;
    case FNC_ROTR:
        if (connectorsDiscartFirst(absoluteConnectors,connection)) { result = layerInputValues[connection] >> (argument1 % BITS_IN_UCHAR); }
        break;
    case FNC_BITSELECTOR:
        if (connectorsDiscartFirst(absoluteConnectors,connection)) { result = layerInputValues[connection] & argument1; }
        break;
    case FNC_SUBS:
        result = argument1;
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result -= layerInputValues[connection]; }
        break;
    case FNC_ADD:
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result += layerInputValues[connection]; }
        break;
    case FNC_MULT:
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result *= layerInputValues[connection]; }
        break;
    case FNC_DIV:
        result = argument1;
        while (connectorsDiscartFirst(absoluteConnectors,connection)) { result /= (max(layerInputValues[connection], (unsigned char) 1)); }
        break;
    case FNC_READX:
        result = pGlobals->testVectors.executionInputLayer[argument1 % pGlobals->settings->circuit.sizeInputLayer];
        break;
    case FNC_EQUAL:
        if (connectorsDiscartFirst(absoluteConnectors,connection)) {
            if (argument1 == layerInputValues[connection]) result = UCHAR_MAX;
        }
        break;

        // unknown function constant
    default:
        mainLogger.out(LOGGER_ERROR) << "Unknown circuit function (" << nodeGetFunction(function) << ")." << endl;
        return STAT_CIRCUIT_INCONSISTENT;
    }
    return STAT_OK;
}
