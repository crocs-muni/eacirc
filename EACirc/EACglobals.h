#ifndef EACGLOBALS_H
#define EACGLOBALS_H

#include "EACconstants.h"
#include "status.h"
//#include "estream/estreamInterface.h"
#include "Logger.h"
//#include "random_generator/IRndGen.h"
class IRndGen;

#include <list>
#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>

//libinclude (galib/GAGenome.h)
#include "GAGenome.h"
//libinclude (galib/GASStateGA.h)
#include "GASStateGA.h"
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


// GENERAL LOGGING SERVICE
/* using main EACirc logger
 *
 * send logs to 'mainLogger.out()' via '<<'
 * flushing 'mainLogger.out()' causes written data to be prefixed by current time and flushed to logging stream
 */
extern Logger mainLogger;
// RANDOM GENERATOR FOR CUSTOM USAGE
/*
 * is initialized at state initialization of EACirc according to external seed or system time
 * idealy should be used ONLY for first initialization of other generators and GAlib
 *
 * internaly is a MD5-based generator
 */
extern IRndGen* mainGenerator;

#define STREAM_BLOCK_SIZE 16

typedef struct _RANDOM_GENERATOR {
    unsigned long       seed;
    bool        useFixedSeed;
	string		QRBGSPath;
    int primaryRandomType;
    int biasRndGenFactor;

    _RANDOM_GENERATOR(void) {
        clear();
    }

    void clear() {
        seed = 0;
        primaryRandomType = 1;
        biasRndGenFactor = 50;
        useFixedSeed = false;
    }

} RANDOM_GENERATOR;

typedef struct _GA_STRATEGY {
    float               probMutationt;
    float               probCrossing;
    int                 popupationSize;
    int                 numGenerations;
    bool                evolutionOff;
    
	_GA_STRATEGY(void) {
        clear();
    }
    
    void clear() {
        probMutationt = 0;
        probCrossing = 0;
        popupationSize = 0;
        numGenerations = 0;
        evolutionOff = false;
    }
} GA_STRATEGY;

typedef struct _GA_CIRCUIT {
    // BASIC CIRCUIT PARAMS
    int         numLayers;  
    int         numSelectorLayers;
    int         sizeInputLayer;
    int         sizeOutputLayer;
    int         sizeLayer;
    int         numConnectors;
    unsigned char        allowedFunctions[FNC_MAX+1];
    int         predictionMethod;
	bool		allowPrunning;
	// TESTING VECTORS PARAMETERS
    int         numTestVectors;
    unsigned char**      testVectors;
    int			changeGalibSeedFrequency; // how often to change GAlib seed and save state
	int			testVectorLength;
    int			saveTestVectors;
    int         testVectorChangeFreq;  // generate fresh new test set every x-th generation
    bool		testVectorChangeProgressive; // change vectors more often in the beginning and less often in the end - use testVectorChangeGener to adjust
	bool		evaluateEveryStep; // evaluation is done only with changing test vectors by default - use with care!
    bool        evaluateBeforeTestVectorChange; // should evaluation before ar after test vectors chagne be written to file?
    int         numBestPredictors;
	bool		representBitAsBytes;
	// SPEED-UP PRECOMPUTATION 
    unsigned long       precompPow[MAX_CONNECTORS];      // PRECOMPUTED VALUES UP TO 2^32
    unsigned long       powEffectiveMask;
	// PARAMETERS OF GENOM FOR THIS CIRCUIT	
	int         genomeSize;
    // INFO ABOUT FITNESS
    float     maxFit;
    float     bestGenerFit;
    float     avgGenerFit;
    int       numAvgGenerFit;
    int       avgPredictions;

    bool      prunningInProgress;

    _GA_CIRCUIT(void) {
        numLayers = MAX_NUM_LAYERS;
        numSelectorLayers = 1;
        sizeInputLayer = MAX_INTERNAL_LAYER_SIZE;
        sizeOutputLayer = MAX_INTERNAL_LAYER_SIZE;
        sizeLayer = MAX_INTERNAL_LAYER_SIZE;
        sizeOutputLayer = MAX_OUTPUTS;
        predictionMethod = 0;
        memset(allowedFunctions, 1, sizeof(allowedFunctions)); // allow all functions by default
        
		allowPrunning = true;
        numTestVectors = 100;
        testVectors = NULL;
		testVectorLength = MAX_INPUTS;
        changeGalibSeedFrequency = 0;
		saveTestVectors = 0;
        testVectorChangeFreq = 0;
        testVectorChangeProgressive = false;
		evaluateEveryStep = false;
        evaluateBeforeTestVectorChange = false;
        numBestPredictors = 1;
		representBitAsBytes = false;

        genomeSize = MAX_GENOME_SIZE;
        
        maxFit = 0;
        bestGenerFit = 0;
        avgGenerFit = 0;
        numAvgGenerFit = 0;
        avgPredictions = 0;
        
        // PRECOMPUTE POW FUNCTION FOR REASONABLE VALUES
        for (int bit = 0; bit < MAX_CONNECTORS; bit++) {
            precompPow[bit] = (unsigned long) pow(2, (float) bit);
            powEffectiveMask |= precompPow[bit];
        }
        
        prunningInProgress = false;
    } 
    
    void allocate() {
        if (testVectors != NULL) release();
        testVectors = new unsigned char*[numTestVectors];
        for (int i = 0; i < numTestVectors; i++) testVectors[i] = new unsigned char[MAX_INPUTS + MAX_OUTPUTS];
    }
    
    void clearFitnessStats() {
        maxFit = 0;
        bestGenerFit = 0;
        avgGenerFit = 0;
        numAvgGenerFit = 0;
        avgPredictions = 0;
    }
    
    void release() {
        if (testVectors != NULL) {
            for (int i = 0; i < numTestVectors; i++) delete[] testVectors[i];
            delete[] testVectors;
            testVectors = NULL;
        }
   }
} GA_CIRCUIT;

typedef struct _BASIC_INIT_DATA {
    string swVersion;
    string computationDate;
    bool recommenceComputation;
    bool loadInitialPopulation;
    int projectType;

    // RANDOM SEED VALUE
    RANDOM_GENERATOR    rndGen;

	// SETTINGS FOR EVOLUTIONARY ALGORITHMS
	GA_STRATEGY			gaConfig;

	// SETTINGS FOR EVOLUTIONARY CIRCUIT
	GA_CIRCUIT			gaCircuitConfig;

    _BASIC_INIT_DATA(void) {
        clear();
    }

    void clear() {
        swVersion = "";
        computationDate = "";
        recommenceComputation = false;
        loadInitialPopulation = false;
        projectType = 0;

        rndGen.clear();
		gaConfig.clear();
    }

} BASIC_INIT_DATA;


#endif //EACGLOBALS_H
