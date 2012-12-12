#ifndef EACGLOBALS_H
#define EACGLOBALS_H

#include "EACconstants.h"
#include "status.h"
#include "estream/estreamInterface.h"
#include "Logger.h"
//#include "random_generator/IRndGen.h"
class IRndGen;

#include <list>
#include <math.h>
//libinclude (galib/GAGenome.h)
#include "GAGenome.h"
//libinclude (galib/GASStateGA.h)
#include "GASStateGA.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
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
    unsigned long       randomSeed;
    bool        useFixedSeed;
	string		QRBGSPath;
	int type;
	int biasFactor;

    _RANDOM_GENERATOR(void) {
        clear();
    }

    void clear() {
        randomSeed = 0;
		type = 1;
		biasFactor = 50;
        useFixedSeed = false;
    }

} RANDOM_GENERATOR;

typedef struct _GA_STRATEGY {
    float               pMutt;
    float               pCross;
    int                 popSize;
    int                 nGeners;
    
	_GA_STRATEGY(void) {
        clear();
    }
    
    void clear() {
        pMutt = 0;
        pCross = 0;
        popSize = 0;
        nGeners = 0;
    }
} GA_STRATEGY;

typedef struct _GA_CIRCUIT {
    // BASIC CIRCUIT PARAMS
    int         numLayers;  
    int         numSelectorLayers;
    int         numInputs;
    int         numOutputs;
    int         internalLayerSize;
    int         outputLayerSize;
    int         numLayerConnectors;
    unsigned char        allowedFNC[FNC_MAX+1];
    int         predictMethod;
	bool		allowPrunning;
	// TESTING VECTORS PARAMETERS
    int         numTestVectors;
    unsigned char**      testVectors;
    int			changeGalibSeedFrequency; // how often to change GAlib seed and save state
    int         testVectorGenerMethod;
	int			testVectorLength;
	int			testVectorBalance;
	int			testVectorEstream;
	int			testVectorEstream2;
	int			estreamKeyType; //what type of data to fill the key/plain/iv
	int			estreamInputType;
	int			estreamIVType;
	int			saveTestVectors;
	int			testVectorEstreamMethod;
	int			limitAlgRoundsCount2;
    int         testVectorChangeGener;  // generate fresh new test set every x-th generation
	bool		TVCGProgressive; // change vectors more often in the beginning and less often in the end - use testVectorChangeGener to adjust
	bool		evaluateEveryStep; // evaluation is done only with changing test vectors by default - use with care!
    int         numBestPredictors;
	bool		limitAlgRounds;
	int			limitAlgRoundsCount;
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
        numInputs = MAX_INTERNAL_LAYER_SIZE;
        numOutputs = MAX_INTERNAL_LAYER_SIZE;
        internalLayerSize = MAX_INTERNAL_LAYER_SIZE;
        outputLayerSize = MAX_OUTPUTS;
        predictMethod = 0;
        memset(allowedFNC, 1, sizeof(allowedFNC)); // allow all functions by default
        
		allowPrunning = true;
        numTestVectors = 100;
        testVectors = NULL;
        testVectorGenerMethod = 0;
		testVectorLength = MAX_INPUTS;
		testVectorBalance = 0;
		testVectorEstream = 0;
		testVectorEstream2 = 0;
		testVectorEstreamMethod = 0;
        changeGalibSeedFrequency = 0;
		estreamKeyType = ESTREAM_GENTYPE_ZEROS;
		estreamInputType = ESTREAM_GENTYPE_ZEROS;
		estreamIVType = ESTREAM_GENTYPE_ZEROS;
		saveTestVectors = 0;
        testVectorChangeGener = 0;
		TVCGProgressive = false;
		evaluateEveryStep = false;
        numBestPredictors = 1;
		limitAlgRounds = false;
		limitAlgRoundsCount = -1;
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
    string simulSWVersion;
    string simulDate;
    bool loadState;

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
        simulSWVersion = "";
        simulDate = "";
        loadState = false;

        rndGen.clear();
		gaConfig.clear();
    }

} BASIC_INIT_DATA;


#endif //EACGLOBALS_H
