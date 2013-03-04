#ifndef EACGLOBALS_H
#define EACGLOBALS_H

#include "EACconstants.h"
#include "Status.h"
#include "Logger.h"
class IRndGen;
//#include "generators/IRndGen.h"
#include <cmath>
#include <cstring>
using namespace std;

// forward declarations
struct SETTINGS_INFO;
struct SETTINGS_MAIN;
struct SETTINGS_RANDOM;
struct SETTINGS_GA;
struct SETTINGS_CIRCUIT;
struct SETTINGS_TEST_VECTORS;
struct SETTINGS;
struct STATISTICS;
struct GLOBALS;

// declarations of global variables (definitions in EACglobals.cpp)
/** main EACirc logging service
  * - send logs to 'mainLogger.out()' via '<<'
  * - flushing 'mainLogger.out()' causes written data to be prefixed by current time and flushed to logging stream
  */
extern Logger mainLogger;

/** main random generator
  * - initialized at state initialization of EACirc according to external seed or system time
  * - idealy should be used ONLY for first initialization of other generators, GAlib, projects, etc.
  * - MD5-based generator internally
  */
extern IRndGen* mainGenerator;

/** random generator for galib seeding
  * - initialized at state initialization of EACirc
  * - used ONLY for GA stuff (GAlib seeding, genome initialization, mutation, crossover)
  * - NEVER use for other purposes
  * - unbiased generator (quantum or MD5-based)
  */
extern IRndGen* galibGenerator;

/** unbiased random generator
  * - initialized at state initialization of EACirc
  * - all-purpose random generator (unbiased)
  * - to be used by 'project' modules (test vector generation)
  */
extern IRndGen* rndGen;

/** biased random generator
  * - initialized at state initialization of EACirc
  * - all-purpose random generator (biased)
  * - to be used by 'project' modules (test vector generation)
  */
extern IRndGen* biasRndGen;

/** globally accessible memory
  * - EACirc settings
  * - test vectors
  * - surrent statistics
  */
extern GLOBALS* pGlobals;

//! settings corresponding to EACIRC/INFO
struct SETTINGS_INFO {
    string swVersion;               //! EACirc framework version
    string computationDate;         //! date of computation
    string notes;                   //! user defined notes
    SETTINGS_INFO();
};

//! settings corresponding to EACIRC/MAIN
struct SETTINGS_MAIN {
    int projectType;                //! project used to generate test vectors
    int evaluatorType;              //! evaluator used in fitness computation
    bool recommenceComputation;     //! is this continuation of previous computation?
    bool loadInitialPopulation;     //! should initial population be loaded instead of randomly generated?
    int numGenerations;             //! number of generations to evolve
    int saveStateFrequency;         //! frequency of reseeding GAlib and saving state
    SETTINGS_MAIN();
};

//! settings corresponding to EACIRC/RANDOM
struct SETTINGS_RANDOM {
    bool useFixedSeed;              //! should computation start from fixed seed instead of generating one?
    unsigned long seed;             //! seed to start from
    int biasRndGenFactor;           //! bias factor for general bias generator
    string qrngPath;                //! path to pregenerated quantum random data
    int qrngFilesMaxIndex;          //! maximal index of qrng data file
    SETTINGS_RANDOM();
};

//! settings corresponding to EACIRC/GA
struct SETTINGS_GA {
    bool evolutionOff;              //! should evolution be turned off?
    float probMutation;             //! probability of genome mutation
    float probCrossing;             //! proprability of genome crossing
    int popupationSize;             //! number of individuals in population
    SETTINGS_GA();
};

//! settings corresponding to EACIRC/CIRCUIT
struct SETTINGS_CIRCUIT {
    int genomeSize;                 //! size of individual genome
    int numLayers;                  //! number of layers in circuit
    int numSelectorLayers;          //! number of input layers
    int sizeLayer;                  //! general layer size
    int sizeInputLayer;             //! number if inputs
    int sizeOutputLayer;            //! number of outputs
    int numConnectors;              //! how many connectors (? TBD)
    bool allowPrunning;             //! allow prunning when writing circuit?
    unsigned char allowedFunctions[FNC_MAX+1];  //! functions allowed in circuit
    SETTINGS_CIRCUIT();
};

//! settings corresponding to EACIRC/TEST_VECTORS
struct SETTINGS_TEST_VECTORS {
    int testVectorLength;                   //! test vector length (in bytes)
    int numTestVectors;                     //! number of test vectors in a testing set
    int testVectorChangeFreq;               //! how often to re-generate test vectors?
    bool testVectorChangeProgressive;       //! change vectors more often in the beginning and less often in the end
    bool saveTestVectors;                   //! should test vecotrs be saved?
    bool evaluateBeforeTestVectorChange;    //! should evaluation before or after test vectors change be written to file?
    bool evaluateEveryStep;                 //! evaluate every step
    // EXTRA INFORMATION, keep updated!
    int numTestSets;                        //! total number of test sets
    SETTINGS_TEST_VECTORS();
};

//! all program run settings
struct SETTINGS {
    SETTINGS_INFO info;
    SETTINGS_MAIN main;
    SETTINGS_RANDOM random;
    SETTINGS_GA ga;
    SETTINGS_CIRCUIT circuit;
    SETTINGS_TEST_VECTORS testVectors;
    void* project;
    SETTINGS();
};

struct STATISTICS {
    int numBestPredictors;          //! TBD
    float maxFit;                   //! TBD
    float bestGenerFit;             //! TBD
    float avgGenerFit;              //! TBD
    int numAvgGenerFit;             //! TBD
    int avgPredictions;             //! TBD
    bool prunningInProgress;        //! is prunning currently in progress?
    STATISTICS();
    void clear();
};

struct GLOBALS {
    SETTINGS* settings;                         //! pointer to SETTINGS in EACirc object
    STATISTICS stats;                           //! current run statistics
    unsigned char** testVectors;                //! current test vector set
    unsigned long precompPow[MAX_CONNECTORS];   //! precomputed values up to 2^32
    unsigned long powEffectiveMask;             //! TBD
    GLOBALS();
    void allocate();
    void release();
};



#endif //EACGLOBALS_H
