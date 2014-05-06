#include "EACglobals.h"

// global variable definitions
Logger mainLogger;
IRndGen* mainGenerator = NULL;
IRndGen* rndGen = NULL;
IRndGen* biasRndGen = NULL;
GLOBALS* pGlobals = NULL;

SETTINGS_MAIN::SETTINGS_MAIN() {
    circuitType = -1;
    projectType = -1;
    evaluatorType = -1;
    evaluatorPrecision = -1;
    recommenceComputation = false;
    loadInitialPopulation = false;
    numGenerations = -1;
    saveStateFrequency = 0;
    circuitSizeInput = -1;
    circuitSizeOutput = -1;
}

SETTINGS_OUTPUTS::SETTINGS_OUTPUTS() {
    graphFiles = true;
    intermediateCircuits = false;
    allowPrunning = false;
    saveTestVectors = false;
}

SETTINGS_RANDOM::SETTINGS_RANDOM() {
    useFixedSeed = false;
    seed = 0;
    biasRndGenFactor = 50;
    useNetShare = false;
    qrngPath = "";
    qrngFilesMaxIndex = -1;
}

SETTINGS_CUDA::SETTINGS_CUDA() {
    enabled = false;
    something = "";
}

SETTINGS_GA::SETTINGS_GA() {
    evolutionOff = false;
    probMutation = -1;
    probCrossing = -1;
    popupationSize = -1;
    replacementSize = -1;
    mutateFunctions = true;
    mutateConnectors = true;
}

SETTINGS_GATE_CIRCUIT::SETTINGS_GATE_CIRCUIT() {
    numLayers = -1;
    sizeLayer = -1;
    useMemory = false;
    sizeMemory = -1;
    numConnectors = -1;
    memset(allowedFunctions, 0, sizeof(allowedFunctions));
    // computed data
    sizeOutputLayer = -1;
    sizeInputLayer = -1;
    genomeSize = -1;
    genomeWidth = -1;
}

SETTINGS_TEST_VECTORS::SETTINGS_TEST_VECTORS() {
    inputLength = -1;
    outputLength = -1;
    setSize = -1;
    setChangeFrequency = -1;
    evaluateBeforeTestVectorChange = false;
    evaluateEveryStep = false;
    numTestSets = -1;
}

SETTINGS_POLY_CIRCUIT::SETTINGS_POLY_CIRCUIT() {
    numPolynomials = -1;
    genomeInitTermStopProbability = -1;
    genomeInitMaxTerms = -1;
    mutateTermStrategy=-1;
    crossoverRandomizePolySelect = false;
    crossoverTermsProbability = -1;
    mutateAddTermProbability = -1;
    mutateAddTermStrategy = -1;
    mutateRemoveTermProbability = -1;
    mutateRemoveTermStrategy = -1;
}

SETTINGS::SETTINGS() {
    project = NULL;
}

STATISTICS::STATISTICS() {
    avgMaxFitSum = 0;
    avgAvgFitSum = 0;
    avgMinFitSum = 0;
    avgCount = 0;
    prunningInProgress = false;
    pvaluesBestIndividual = NULL;
}

TEST_VECTORS::TEST_VECTORS() {
    inputs = NULL;
    outputs = NULL;
    circuitOutputs = NULL;
    newSet = false;
    executionInputLayer = NULL;
    executionMiddleLayerIn = NULL;
    executionMiddleLayerOut = NULL;
    executionOutputLayer = NULL;
}

void TEST_VECTORS::allocate() {
    if (pGlobals->settings->testVectors.inputLength == -1 || pGlobals->settings->testVectors.outputLength == -1
            || pGlobals->settings->main.circuitSizeOutput == -1) {
        mainLogger.out(LOGGER_ERROR) << "Test vector input/output size or circuit output size not set." << endl;
        return;
    }
    // if memory is allocated, release
    if (inputs != NULL || outputs != NULL || circuitOutputs != NULL || executionInputLayer != NULL
            || executionMiddleLayerIn != NULL || executionMiddleLayerOut != NULL || executionOutputLayer != NULL) release();
    // allocate memory for inputs, outputs, citcuitOutputs
    inputs = new unsigned char*[pGlobals->settings->testVectors.setSize];
    outputs = new unsigned char*[pGlobals->settings->testVectors.setSize];
    circuitOutputs = new unsigned char*[pGlobals->settings->testVectors.setSize];
    for (int i = 0; i < pGlobals->settings->testVectors.setSize; i++) {
        inputs[i] = new unsigned char[pGlobals->settings->testVectors.inputLength];
        memset(inputs[i],0,pGlobals->settings->testVectors.inputLength);
        outputs[i] = new unsigned char[pGlobals->settings->testVectors.outputLength];
        memset(outputs[i],0,pGlobals->settings->testVectors.outputLength);
        circuitOutputs[i] = new unsigned char[pGlobals->settings->main.circuitSizeOutput];
        memset(circuitOutputs[i],0,pGlobals->settings->main.circuitSizeOutput);
    }
    
    if (pGlobals->settings->gateCircuit.sizeInputLayer <=0 
            || pGlobals->settings->gateCircuit.sizeLayer <= 0
            || pGlobals->settings->gateCircuit.sizeOutputLayer <= 0){
        mainLogger.out(LOGGER_INFO) << "TestVectors: no memory allocated for gate related representation. Invalid dimensions." << endl;
    } else {
        executionInputLayer = new unsigned char[pGlobals->settings->gateCircuit.sizeInputLayer];
        executionMiddleLayerIn = new unsigned char[pGlobals->settings->gateCircuit.sizeLayer];
        executionMiddleLayerOut = new unsigned char[pGlobals->settings->gateCircuit.sizeLayer];
        executionOutputLayer = new unsigned char[pGlobals->settings->gateCircuit.sizeOutputLayer];
    }
}

void TEST_VECTORS::release() {
    if (inputs != NULL) {
        for (int i = 0; i < pGlobals->settings->testVectors.setSize; i++) delete[] inputs[i];
        delete[] inputs;
        inputs = NULL;
    }
    if (outputs != NULL) {
        for (int i = 0; i < pGlobals->settings->testVectors.setSize; i++) delete[] outputs[i];
        delete[] outputs;
        outputs = NULL;
    }
    if (circuitOutputs != NULL) {
        for (int i = 0; i < pGlobals->settings->testVectors.setSize; i++) delete[] (circuitOutputs[i]);
        delete[] circuitOutputs;
        circuitOutputs = NULL;
    }
    if (executionInputLayer != NULL) delete[] executionInputLayer;
    executionInputLayer = NULL;
    if (executionMiddleLayerIn != NULL) delete[] executionMiddleLayerIn;
    executionMiddleLayerIn = NULL;
    if (executionMiddleLayerOut != NULL) delete[] executionMiddleLayerOut;
    executionMiddleLayerOut = NULL;
    if (executionOutputLayer != NULL) delete[] executionOutputLayer;
    executionOutputLayer = NULL;
}

void STATISTICS::allocate() {
    if (pvaluesBestIndividual!=NULL){
        release();
    }
    
    pvaluesBestIndividual = new vector<double>;
}

void STATISTICS::release() {
    if (pvaluesBestIndividual!=NULL) delete pvaluesBestIndividual;
    pvaluesBestIndividual = NULL;
}

GLOBALS::GLOBALS() {
    settings = NULL;
    evaluator = NULL;
    // precompute powers for reasonable values (2^0-2^31)
    for (int bit = 0; bit < MAX_LAYER_SIZE; bit++) {
        precompPow[bit] = (unsigned long) pow(2, (float) bit);
    }
}
