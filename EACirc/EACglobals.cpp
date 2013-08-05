#include "EACglobals.h"

// global variable definitions
Logger mainLogger;
IRndGen* mainGenerator = NULL;
IRndGen* rndGen = NULL;
IRndGen* biasRndGen = NULL;
GLOBALS* pGlobals = NULL;

SETTINGS_MAIN::SETTINGS_MAIN() {
    projectType = -1;
    evaluatorType = -1;
    evaluatorPrecision = -1;
    recommenceComputation = false;
    loadInitialPopulation = false;
    numGenerations = -1;
    saveStateFrequency = 0;
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
    mutateFunctions = true;
    mutateConnectors = true;
}

SETTINGS_CIRCUIT::SETTINGS_CIRCUIT() {
    numLayers = -1;
    sizeLayer = -1;
    sizeInputLayer = -1;
	sizeOutputLayer = -1;
    totalSizeOutputLayer = -1;
    numConnectors = -1;
    useMemory = false;
    memorySize = -1;
    memset(allowedFunctions, 0, sizeof(allowedFunctions));
    genomeSize = -1;
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

SETTINGS::SETTINGS() {
    project = NULL;
}

STATISTICS::STATISTICS() {
    avgMaxFitSum = 0;
    avgAvgFitSum = 0;
    avgMinFitSum = 0;
    avgCount = 0;
    prunningInProgress = false;
}

TEST_VECTORS::TEST_VECTORS() {
    inputs = NULL;
    outputs = NULL;
    circuitOutputs = NULL;
    newSet = false;
}

void TEST_VECTORS::allocate() {
    if (pGlobals->settings->testVectors.inputLength == -1 || pGlobals->settings->testVectors.outputLength == -1
            || pGlobals->settings->circuit.totalSizeOutputLayer == -1) {
        mainLogger.out(LOGGER_ERROR) << "Test vector input/output size or circuit output size not set." << endl;
        return;
    }
    // if memory is allocated, release
    if (inputs != NULL || outputs != NULL || circuitOutputs != NULL) release();
    // allocate memory for inputs, outputs, citcuitOutputs
    inputs = new unsigned char*[pGlobals->settings->testVectors.setSize];
    outputs = new unsigned char*[pGlobals->settings->testVectors.setSize];
    circuitOutputs = new unsigned char*[pGlobals->settings->testVectors.setSize];
    for (int i = 0; i < pGlobals->settings->testVectors.setSize; i++) {
        inputs[i] = new unsigned char[pGlobals->settings->testVectors.inputLength];
        memset(inputs[i],0,pGlobals->settings->testVectors.inputLength);
        outputs[i] = new unsigned char[pGlobals->settings->testVectors.outputLength];
        memset(outputs[i],0,pGlobals->settings->testVectors.outputLength);
        circuitOutputs[i] = new unsigned char[pGlobals->settings->circuit.totalSizeOutputLayer];
        memset(circuitOutputs[i],0,pGlobals->settings->circuit.totalSizeOutputLayer);
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
}

GLOBALS::GLOBALS() {
    settings = NULL;
    evaluator = NULL;
    // precompute powers for reasonable values (2^0-2^31)
    for (int bit = 0; bit < MAX_LAYER_SIZE; bit++) {
        precompPow[bit] = (unsigned long) pow(2, (float) bit);
        powEffectiveMask |= precompPow[bit];
    }
}
