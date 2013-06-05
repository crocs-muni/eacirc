#ifndef EACGLOBALS_H
#define EACGLOBALS_H

#include "EACconstants.h"
#include "Status.h"
#include "Logger.h"
class IEvaluator;
//#include "evaluators/IEvaluator.h"
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

/**
  Type used to store basic genom items - connectors and functions
*/
typedef unsigned long GENOM_ITEM_TYPE;

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
    bool useNetShare;               //! try to map net share (used on METACENTRUM resources)
    string qrngPath;                //! path to pregenerated quantum random data
    int qrngFilesMaxIndex;          //! maximal index of qrng data file
    SETTINGS_RANDOM();
};

//! settings corresponding to EACIRC/CUDA
struct SETTINGS_CUDA {
    bool enabled;                   //! is CUDA support enabled?
    string something;               //! string setting example
    SETTINGS_CUDA();
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
    int numLayers;                  //! number of layers in circuit
    int sizeLayer;                  //! general layer size
    int sizeInputLayer;             //! number if inputs
    int sizeOutputLayer;            //! number of outputs
    int totalSizeOutputLayer;       //! number of outputs (including possible memory outputs)
    int numConnectors;              //! how many connectors (? TBD)
    bool useMemory;                 //! should we return part of output to input as memory?
    int memorySize;                 //! memory size in bytes
    bool allowPrunning;             //! allow prunning when writing circuit?
    unsigned char allowedFunctions[FNC_MAX+1];  //! functions allowed in circuit
    // EXTRA INFORMATION, keep updated!
    int genomeSize;                 //! size of individual genome
    SETTINGS_CIRCUIT();
};

//! settings corresponding to EACIRC/TEST_VECTORS
struct SETTINGS_TEST_VECTORS {
    int inputLength;                        //! test vector length (input for circuit) (in bytes)
    int outputLength;                       //! expected test vector output length (output from circuit) (in bytes)
    int setSize;                            //! number of test vectors in a testing set
    int setChangeFrequency;                 //! how often to re-generate test vectors?
    bool setChangeProgressive;              //! change vectors more often in the beginning and less often in the end
    bool saveTestVectors;                   //! should test vecotrs be saved?
    bool evaluateBeforeTestVectorChange;    //! should evaluation before or after test vectors change be written to file?
    bool evaluateEveryStep;                 //! evaluate every step
    // EXTRA INFORMATION, keep updated!
    int numTestSets;                        //! total number of test sets
    SETTINGS_TEST_VECTORS();
};

//! all program run settings
struct SETTINGS {
    SETTINGS_INFO info;                     //! corresponding to EACIRC/INFO
    SETTINGS_MAIN main;                     //! corresponding to EACIRC/MAIN
    SETTINGS_RANDOM random;                 //! corresponding to EACIRC/RANDOM
    SETTINGS_CUDA cuda;                     //! corresponding to EACIRC/CUDA
    SETTINGS_GA ga;                         //! corresponding to EACIRC/GA
    SETTINGS_CIRCUIT circuit;               //! corresponding to EACIRC/CIRCUIT
    SETTINGS_TEST_VECTORS testVectors;      //! corresponding to EACIRC/TEST_VECTORS
    void* project;                          //! project specific settings
    SETTINGS();
};

//! main fitness statistics
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

//! test vectors and their outputs
struct TEST_VECTORS {
    unsigned char** inputs;             //! test vector inputs for current set
    unsigned char** outputs;            //! (correct) test vector outputs for current set
    unsigned char** circuitOutputs;     //! circuit outputs for current set (to ease memory allocation)
    double	circuitOutputCategories[NUM_OUTPUT_CATEGORIES];	//! frequency of occurence of output categories as classified by the circuit
    double	circuitOutputCategoriesRandom[NUM_OUTPUT_CATEGORIES];	//! frequency of occurence of output categories as classified by the circuit provided with truly random data

    bool newSet;                        //! has new set been generated? (for CUDA usage)
    TEST_VECTORS();
    void allocate();
    void release();
};

//! globally accessible data (settings, stats, test vectors)
struct GLOBALS {
    SETTINGS* settings;                         //! pointer to SETTINGS in EACirc object
    STATISTICS stats;                           //! current run statistics
    TEST_VECTORS testVectors;                   //! current test vector set
    IEvaluator* evaluator;                      //! evaluator (compares expected output with actual circuit output)
    unsigned long precompPow[MAX_LAYER_SIZE];   //! precomputed values up to 2^32
    unsigned long powEffectiveMask;             //! TBD
    GLOBALS();
};

#endif //EACGLOBALS_H
