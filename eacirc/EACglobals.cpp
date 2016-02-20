#include "EACglobals.h"

// global variable definitions
Logger mainLogger;
IRndGen* mainGenerator = NULL;
IRndGen* rndGen = NULL;
IRndGen* biasRndGen = NULL;
IRndGen* lutRndGen = NULL;
GLOBALS* pGlobals = NULL;

SETTINGS_MAIN::SETTINGS_MAIN() {
    circuitType = -1;
    projectType = -1;
    evaluatorType = -1;
    evaluatorPrecision = -1;
    significanceLevel = 5;
    recommenceComputation = false;
    loadInitialPopulation = false;
    numGenerations = -1;
    saveStateFrequency = 0;
    circuitSizeInput = -1;
    circuitSizeOutput = -1;
}

SETTINGS_OUTPUTS::SETTINGS_OUTPUTS() {
    verbosity = 0;
    intermediateCircuits = false;
    allowPrunning = false;
    saveTestVectors = false;
    fractionFile = false;
}

SETTINGS_RANDOM::SETTINGS_RANDOM() {
    useFixedSeed = false;
    seed = 0;
    biasRndGenFactor = 50;
	lutHW = 262144;
    useNetShare = false;
    qrngPath = "";
    qrngFilesMaxIndex = -1;
}

SETTINGS_CUDA::SETTINGS_CUDA() {
    enabled = false;
    block_size = 512;
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
    jvmSim = NULL;
}

SETTINGS_TEST_VECTORS::SETTINGS_TEST_VECTORS() {
	generator = 0;
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
    genomeInitTermCountProbability = -1;
    maxNumTerms = -1;
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
    actGener = 0;
    avgMaxFitSum = 0;
    avgAvgFitSum = 0;
    avgMinFitSum = 0;
    avgCount = 0;
    prunningInProgress = false;
    pvaluesBestIndividual = NULL;
}

void TEST_VECTORS::allocate() {
    if (pGlobals->settings->testVectors.inputLength == -1 || pGlobals->settings->testVectors.outputLength == -1
            || pGlobals->settings->main.circuitSizeOutput == -1) {
        mainLogger.out(LOGGER_ERROR) << "Test vector input/output size or circuit output size not set." << endl;
        return;
    }

    inputs = TestVectors( pGlobals->settings->testVectors.inputLength, pGlobals->settings->testVectors.setSize );
    outputs = TestVectors( pGlobals->settings->testVectors.outputLength, pGlobals->settings->testVectors.setSize );
    circuitOutputs = TestVectors( pGlobals->settings->testVectors.outputLength, pGlobals->settings->testVectors.setSize );
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
    circuit = NULL;
    // precompute powers for reasonable values (2^0-2^31)
    for (int bit = 0; bit < MAX_LAYER_SIZE; bit++) {
        precompPow[bit] = (unsigned long) pow(2, (float) bit);
    }
}
