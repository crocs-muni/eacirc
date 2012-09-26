#ifndef SSGLOBALS_H
#define SSGLOBALS_H

#include "SSconstants.h"
#include "status.h"
#include "estream/estream-interface.h"

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

#ifdef _MSC_VER

typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;

#else
#include <stdint.h>
#endif


// GENERAL LOGGING SERVICE
#define LOG_INSERTSTRING(message)       { ofstream out("output.log", fstream::app); out << message << endl; out.close();}

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

// GA CIRCUIT CONSTANTS
#define NUM_BITS                    8             // NUMBER OF BITS PER unsigned char
#define NUM_TEST_SETS               1000
#define MAX_TEST_SET_SIZE_PER_GENOM 1000   
#define TEST_SET_SIZE               MAX_TEST_SET_SIZE_PER_GENOM * NUM_TEST_SETS

#define PREDICT_BIT                 0
#define PREDICT_BITGROUP_PARITY     1
#define PREDICT_BYTES_PARITY        2
#define PREDICT_HAMMING_WEIGHT      3
#define PREDICT_BYTE				4
#define	PREDICT_DISTINGUISH			5
#define	PREDICT_AVALANCHE			6

#define TESTVECT_MD5INV             0
#define TESTVECT_SHA1INV            1
#define TESTVECT_DESPLAINTEXT       2
#define TESTVECT_TEST               3
#define TESTVECT_MD5SHA_DISTINGUISH		4
#define TESTVECT_MD5_RAND_DISTINGUISH   5
#define TESTVECT_SHA1_RAND_DISTINGUISH  6

//RNDGEN CONSTANTS
#define CRNDGEN						1
#define	BIASGEN						2

#define FNC_NOP                     0
#define FNC_OR                      1
#define FNC_AND                     2
#define FNC_CONST                   3
#define FNC_XOR                     4
#define FNC_NOR                     5
#define FNC_NAND                    6
#define FNC_ROTL                    7
#define FNC_ROTR                    8
#define FNC_BITSELECTOR             9
#define FNC_SUM                     10
#define FNC_SUBS                    11
#define FNC_ADD                     12
#define FNC_MULT                    13
#define FNC_DIV                     14
#define FNC_READX                   15
#define FNC_MAX                     FNC_READX

#define MAX_NUM_LAYERS              100
#define MAX_CONNECTORS              32
#define MAX_INTERNAL_LAYER_SIZE     32
#define MAX_OUTPUTS                 MAX_INTERNAL_LAYER_SIZE  
#define MAX_INPUTS                  MAX_INTERNAL_LAYER_SIZE * MAX_INTERNAL_LAYER_SIZE // IF bSectorData IS ENABLED, SEPARATE RANGE FOR EACH INPUT NODE IN FIRST LAYER IS USED  
#define MAX_GENOME_SIZE (MAX_NUM_LAYERS * 2) * MAX_INTERNAL_LAYER_SIZE

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
	int			testVectorGenerChangeSeed; // whether to change seed every new generation
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
		testVectorGenerChangeSeed = 0;
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

        rndGen.clear();
		gaConfig.clear();
    }

} BASIC_INIT_DATA;


#endif
