#include "EACglobals.h"

// global variable definitions
Logger mainLogger;
IRndGen* mainGenerator = NULL;
IRndGen* galibGenerator = NULL;
IRndGen* rndGen = NULL;
IRndGen* biasRndGen = NULL;
GLOBALS* pGlobals = NULL;

SETTINGS_INFO::SETTINGS_INFO() {
    swVersion="unknown";
    computationDate="unknown";
    notes = "";
}

SETTINGS_MAIN::SETTINGS_MAIN() {
    projectType = -1;
    evaluatorType = -1;
    recommenceComputation = false;
    loadInitialPopulation = false;
    numGenerations = -1;
    saveStateFrequency = 0;
}

SETTINGS_RANDOM::SETTINGS_RANDOM() {
    useFixedSeed = false;
    seed = 0;
    biasRndGenFactor = 50;
    qrngPath = "";
    qrngFilesMaxIndex = -1;
}

SETTINGS_GA::SETTINGS_GA() {
    evolutionOff = false;
    probMutation = -1;
    probCrossing = -1;
    popupationSize = -1;
}

SETTINGS_CIRCUIT::SETTINGS_CIRCUIT() {
    genomeSize = -1;
    numLayers = -1;
    numSelectorLayers = 1;
    sizeLayer = -1;
    sizeInputLayer = -1;
    sizeOutputLayer = -1;
    numConnectors = -1;
    memset(allowedFunctions, 1, sizeof(allowedFunctions)); // all allowed by default
}

SETTINGS_TEST_VECTORS::SETTINGS_TEST_VECTORS() {
    inputLength = -1;
    outputLength = -1;
    setSize = -1;
    setChangeFrequency = -1;
    setChangeProgressive = false;
    saveTestVectors = true;
    evaluateBeforeTestVectorChange = false;
    evaluateEveryStep = false;
    numTestSets = -1;
}

SETTINGS::SETTINGS() {
    project = NULL;
}

STATISTICS::STATISTICS() {
    clear();
}

void STATISTICS::clear() {
    numBestPredictors = 0;
    maxFit = 0;
    bestGenerFit = 0;
    avgGenerFit = 0;
    numAvgGenerFit = 0;
    avgPredictions = 0;
    prunningInProgress = false;
}

GLOBALS::GLOBALS() {
    // precompute powers for reasonable values (2^0-2^31)
    for (int bit = 0; bit < MAX_CONNECTORS; bit++) {
        precompPow[bit] = (unsigned long) pow(2, (float) bit);
        powEffectiveMask |= precompPow[bit];
    }
    testVectors = NULL;
}

void GLOBALS::allocate() {
    if (settings->testVectors.inputLength == -1 || settings->testVectors.outputLength == -1) {
        mainLogger.out(LOGGER_ERROR) << "Test vector input/output size not set." << endl;
        return;
    }
    if (testVectors != NULL) release();
    testVectors = new unsigned char*[settings->testVectors.setSize];
    for (int i = 0; i < settings->testVectors.setSize; i++) {
        testVectors[i] = new unsigned char[settings->testVectors.inputLength +
                settings->testVectors.outputLength];
        memset(testVectors[i],0,settings->testVectors.inputLength +
               settings->testVectors.outputLength);
    }
}

void GLOBALS::release() {
    if (testVectors != NULL) {
        for (int i = 0; i < settings->testVectors.setSize; i++) delete[] testVectors[i];
        delete[] testVectors;
        testVectors = NULL;
    }
}
