#ifndef EACGLOBALS_H
#define EACGLOBALS_H

#include "EACconstants.h"
#include "Status.h"
#include "Logger.h"

#include <cmath>
#include <cstring>

using namespace std;

typedef unsigned long DWORD;

#ifndef FALSE
#define FALSE               0
#endif

#ifndef TRUE
#define TRUE                1
#endif

#ifndef ULONG_MAX
#define ULONG_MAX     0xffffffffUL
#endif

#ifndef INT_MAX
#define INT_MAX       2147483647
#endif

#ifdef _MSC_VER
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif

/** main EACirc logging service
  * - send logs to 'mainLogger.out()' via '<<'
  * - flushing 'mainLogger.out()' causes written data to be prefixed by current time and flushed to logging stream
  */
extern Logger mainLogger;

//! settings corresponding to EACIRC/INFO
typedef struct _SETTINGS_INFO {
    string swVersion;               //! EACirc framework version
    string computationDate;         //! date of computation
    string notes;                   //! user defined notes
    _SETTINGS_INFO(void) {
        swVersion="unknown";
        computationDate="unknown";
        notes = "";
    }
} SETTINGS_INFO;

//! settings corresponding to EACIRC/MAIN
typedef struct _SETTINGS_MAIN {
    int projectType;                //! project used to generate test vectors
    int evaluatorType;              //! evaluator used in fitness computation
    bool recommenceComputation;     //! is this continuation of previous computation?
    bool loadInitialPopulation;     //! should initial population be loaded instead of randomly generated?
    int numGenerations;             //! number of generations to evolve
    int saveStateFrequency;         //! frequency of reseeding GAlib and saving state
    _SETTINGS_MAIN(void) {
        projectType = PROJECT_PREGENERATED_TV;
        evaluatorType = EVALUATOR_HAMMING_WEIGHT;
        recommenceComputation = false;
        loadInitialPopulation = false;
        numGenerations = 0;
        saveStateFrequency = 0;
    }
} SETTINGS_MAIN;

//! settings corresponding to EACIRC/RANDOM
typedef struct _SETTINGS_RANDOM {
    bool useFixedSeed;              //! should computation start from fixed seed instead of generating one?
    unsigned long seed;             //! seed to start from
//    int primaryRandomType;          //! type of random for each rand()
    int biasRndGenFactor;           //! bias factor for general bias generator
    string qrngPath;                //! path to pregenerated quantum random data
    _SETTINGS_RANDOM(void) {
        useFixedSeed = false;
        seed = 0;
//        primaryRandomType = 1;
        biasRndGenFactor = 50;
        qrngPath = "";
    }
} SETTINGS_RANDOM;

//! settings corresponding to EACIRC/GA
typedef struct _SETTINGS_GA {
    bool evolutionOff;              //! should evolution be turned off?
    float probMutation;             //! probability of genome mutation
    float probCrossing;             //! proprability of genome crossing
    int popupationSize;             //! number of individuals in population
    _SETTINGS_GA(void) {
        evolutionOff = false;
        probMutation = 0;
        probCrossing = 0;
        popupationSize = 0;
    }
} SETTINGS_GA;

//! settings corresponding to EACIRC/CIRCUIT
typedef struct _SETTINGS_CIRCUIT {
    int genomeSize;                 //! size of individual genome
    int numLayers;                  //! number of layers in circuit
    int numSelectorLayers;          //! number of input layers
    int sizeLayer;                  //! general layer size
    int sizeInputLayer;             //! number if inputs
    int sizeOutputLayer;            //! number of outputs
    int numConnectors;              //! how many connectors (? TBD)
    bool allowPrunning;             //! allow prunning when writing circuit?
    unsigned char allowedFunctions[FNC_MAX+1];  //! functions allowed in circuit
    _SETTINGS_CIRCUIT(void) {
        genomeSize = MAX_GENOME_SIZE;
        numLayers = MAX_NUM_LAYERS;
        numSelectorLayers = 1;
        sizeLayer = MAX_INTERNAL_LAYER_SIZE;
        sizeInputLayer = MAX_INTERNAL_LAYER_SIZE;
        sizeOutputLayer = MAX_OUTPUTS;
        numConnectors = 0;
        memset(allowedFunctions, 1, sizeof(allowedFunctions)); // all allowed by default
    }
} SETTINGS_CIRCUIT;

//! settings corresponding to EACIRC/TEST_VECTORS
typedef struct _SETTINGS_TEST_VECTORS {
    int testVectorLength;                   //! test vector length (in bytes)
    int numTestVectors;                     //! number of test vectors in a testing set
    int testVectorChangeFreq;               //! how often to re-generate test vectors?
    bool testVectorChangeProgressive;       //! change vectors more often in the beginning and less often in the end
    bool saveTestVectors;                   //! should test vecotrs be saved?
    bool evaluateBeforeTestVectorChange;    //! should evaluation before or after test vectors change be written to file?
    bool evaluateEveryStep;                 //! evaluate every step
    _SETTINGS_TEST_VECTORS(void) {
        testVectorLength = MAX_INPUTS;
        numTestVectors = 100;
        testVectorChangeFreq = 0;
        testVectorChangeProgressive = false;
        saveTestVectors = true;
        evaluateBeforeTestVectorChange = false;
        evaluateEveryStep = false;
    }
} SETTINGS_TEST_VECTORS;

//! all program run settings
typedef struct _SETTINGS {
    SETTINGS_INFO info;
    SETTINGS_MAIN main;
    SETTINGS_RANDOM random;
    SETTINGS_GA ga;
    SETTINGS_CIRCUIT circuit;
    SETTINGS_TEST_VECTORS testVectors;
    void* project;
    _SETTINGS(void) {
        project = NULL;
    }
} SETTINGS;

typedef struct _STATISTICS {
    int numBestPredictors;          //! TBD
    float maxFit;                   //! TBD
    float bestGenerFit;             //! TBD
    float avgGenerFit;              //! TBD
    int numAvgGenerFit;             //! TBD
    int avgPredictions;             //! TBD
    bool prunningInProgress;        //! is prunning currently in progress?
    _STATISTICS(void) {
        clear();
    }
    void clear() {
        numBestPredictors = 0;
        maxFit = 0;
        bestGenerFit = 0;
        avgGenerFit = 0;
        numAvgGenerFit = 0;
        avgPredictions = 0;
        prunningInProgress = false;
    }
} STATISTICS;

typedef struct _GLOBALS {
    SETTINGS* settings;                         //! pointer to SETTINGS in EACirc object
    STATISTICS stats;                           //! current run statistics
    unsigned char** testVectors;                //! current test vector set
    unsigned long precompPow[MAX_CONNECTORS];   //! precomputed values up to 2^32
    unsigned long powEffectiveMask;             //! TBD
    _GLOBALS(void) {
        // precompute powers for reasonable values (2^0-2^31)
        for (int bit = 0; bit < MAX_CONNECTORS; bit++) {
            precompPow[bit] = (unsigned long) pow(2, (float) bit);
            powEffectiveMask |= precompPow[bit];
        }
        testVectors = NULL;
    }
    void allocate() {
        if (testVectors != NULL) release();
        testVectors = new unsigned char*[settings->testVectors.numTestVectors];
        for (int i = 0; i < settings->testVectors.numTestVectors; i++) testVectors[i] = new unsigned char[MAX_INPUTS + MAX_OUTPUTS];
    }
    void release() {
        if (testVectors != NULL) {
            for (int i = 0; i < settings->testVectors.numTestVectors; i++) delete[] testVectors[i];
            delete[] testVectors;
            testVectors = NULL;
        }
    }
} GLOBALS;

#endif //EACGLOBALS_H
