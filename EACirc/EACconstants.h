#ifndef EACCONSTANTS_H
#define EACCONSTANTS_H

// FILENAMES
#define FILE_CONFIG                 "config.xml"
#define FILE_SEEDFILE               "LastSeed.txt"
#define FILE_FITNESS_PROGRESS       "EAC_fitnessProgress.txt"
#define FILE_BEST_FITNESS           "bestfitgraph.txt"
#define FILE_AVG_FITNESS            "avgfitgraph.txt"
#define FILE_GALIB_SCORES           "scores.log"
#define FILE_BOINC_FRACTION_DONE    "fraction_done.txt"
#define FILE_TEST_VECTORS           "TestVectors.txt"
#define FILE_TEST_DATA_1            "TestData1.txt"
#define FILE_TEST_DATA_2            "TestData2.txt"
//#define ???                         "EAC_circuit.bin"

// QRNG DATA (filename = $PREFIX$INDEX$DUFFIX)
#define	FILE_QRNG_DATA_INDEX_MAX	10
#define FILE_QRNG_DATA_PREFIX       "qrng-"
#define FILE_QRNG_DATA_SUFFIX       ".bin"
// maximum number of bits used from random data file
// don't set too big, also denotes size of created system random (??)
#define RANDOM_DATA_FILE_SIZE		10000000

// COMMAND LINE OPTIONS
#define CMD_OPT_STATIC              "-staticcircuit"
#define CMD_OPT_EVOLUTION_OFF       "-evolutionoff"
#define CMD_OPT_STATIC_DISTINCTOR   "-distinctor"

// GA CIRCUIT CONSTANTS
#define NUM_BITS                    8             // NUMBER OF BITS PER unsigned char
#define NUM_TEST_SETS               1000
#define MAX_TEST_SET_SIZE_PER_GENOM 1000
#define TEST_SET_SIZE               MAX_TEST_SET_SIZE_PER_GENOM * NUM_TEST_SETS

// CIRCUIT LIMITS
#define MAX_NUM_LAYERS              100
#define MAX_CONNECTORS              32
#define MAX_INTERNAL_LAYER_SIZE     32
#define MAX_OUTPUTS                 MAX_INTERNAL_LAYER_SIZE
#define MAX_INPUTS                  MAX_INTERNAL_LAYER_SIZE * MAX_INTERNAL_LAYER_SIZE // IF bSectorData IS ENABLED, SEPARATE RANGE FOR EACH INPUT NODE IN FIRST LAYER IS USED
#define MAX_GENOME_SIZE (MAX_NUM_LAYERS * 2) * MAX_INTERNAL_LAYER_SIZE

// PREDICTORS
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

// RANDOM GENERATORS
#define CRNDGEN						1
#define	BIASGEN						2

// CIRCUIT FUNCTIONS
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

#endif                                                       