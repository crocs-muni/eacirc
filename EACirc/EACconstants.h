#ifndef EACCONSTANTS_H
#define EACCONSTANTS_H

#include <limits.h>

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

#ifndef UCHAR_MAX
#define UCHAR_MAX     255
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

// PROJECTS
#define PROJECT_PREGENERATED_TV     100
#define PROJECT_ESTREAM             101
#define PROJECT_SHA3                102

// FILENAMES
#define FILE_CONFIG                 "config.xml"
#define FILE_STATE                  "state.xml"
#define FILE_STATE_INITIAL          "state_initial.xml"
#define FILE_POPULATION             "population.xml"
#define FILE_POPULATION_INITIAL     "population_initial.xml"
#define FILE_FITNESS_PROGRESS       "fitness_progress.txt"
#define FILE_BEST_FITNESS           "bestfit_graph.txt"
#define FILE_AVG_FITNESS            "avgfit_graph.txt"
#define FILE_GALIB_SCORES           "scores.log"
#define FILE_BOINC_FRACTION_DONE    "fraction_done.txt"
#define FILE_TEST_VECTORS           "test_vectors.txt"
#define FILE_TEST_VECTORS_HR        "test_vectors_hr.txt"
#define FILE_TEST_DATA_1            "test_vecotrs1.bin"
#define FILE_TEST_DATA_2            "test_vectors2.bin"
#define FILE_CIRCUIT                "circuit_"          // folloved by fitness (.bin, .txt, .dot, .c)
#define FILE_BEST_CIRCUIT           "EAC_circuit"       // .bin, .txt, .dot, .c
#define FILE_LOGFILE                "eacirc.log"

// CIRCUIT OUTPUT SETTINGS
#define CIRCUIT_FILENAME_PRECISION 3

// QRNG DATA
// filename = $PREFIX$INDEX$DUFFIX
// index starts from 0 to maximal value save in main settings
// index number should be prefixed by zeroes (to equal length as max index)
#define FILE_QRNG_DATA_PREFIX       "Random"
#define FILE_QRNG_DATA_SUFFIX       ".bin"
// maximum number of bits used from random data file
// don't set too big, this size is read into memory
// currently 10MB
#define RANDOM_DATA_FILE_SIZE		10485760

// COMMAND LINE OPTIONS
#define CMD_OPT_STATIC              "-staticcircuit"
#define CMD_OPT_STATIC_DISTINCTOR   "-distinctor"
#define CMD_OPT_LOGGING             "-log"
#define CMD_OPT_LOGGING_TO_FILE     "-log2file"
#define CMD_OPT_SELF_TEST           "-test"

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

// EVALUATORS
#define EVALUATOR_BIT               0
#define EVALUATOR_BITGROUP_PARITY   1
#define EVALUATOR_BYTES_PARITY      2
#define EVALUATOR_HAMMING_WEIGHT    3
#define EVALUATOR_BYTE              4
#define EVALUATOR_DISTINGUISH       5
#define EVALUATOR_AVALANCHE         6

#define TESTVECT_MD5INV             0
#define TESTVECT_SHA1INV            1
#define TESTVECT_DESPLAINTEXT       2
#define TESTVECT_TEST               3
#define TESTVECT_MD5SHA_DISTINGUISH		4
#define TESTVECT_MD5_RAND_DISTINGUISH   5
#define TESTVECT_SHA1_RAND_DISTINGUISH  6

// RANDOM GENERATORS
#define GENERATOR_QRNG				1
#define	GENERATOR_BIAS				2
#define GENERATOR_MD5               3

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
