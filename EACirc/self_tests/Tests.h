#ifndef TESTS_H
#define TESTS_H

#include <string>
using std::string;
#include "Catch.h"
#include "EACglobals.h"

#define BACKUP_SUFFIX ".2"

/** compare contents of given files line by line
  * if files differ in more than 5 lines, comparation is terminated
  *
  * @param filename1
  * @param filename2
  */
void compareFilesByLine(string filename1, string filename2);

/** backup given file (rename with BACKUP-SUFFIX)
  * any previous version of backup-file is overwritten
  *
  * @param filename
  * @return status
  */
int backupFile(string filename);

/** backup common result files
  * FILE_GALIB_SCORES, FILE_FITNESS_PROGRESS, FILE_BEST_FITNESS, FILE_AVG_FITNESS, FILE_STATE, FILE_POPULATION
  */
void backupResults();

/** compare common result files with their backuped versions
  * FILE_GALIB_SCORES, FILE_FITNESS_PROGRESS, FILE_BEST_FITNESS, FILE_AVG_FITNESS, FILE_STATE, FILE_POPULATION
  */
void compareResults();

/** run EACirc computation
  * perform all nedded steps (create, load, initialize, prepare, run)
  *
  * @return status from finished EACirc run
  */
int runEACirc();

// to be moved into Project class (each project will be supposed to provide a string with basic configuration)
// hm, automatic determinism testing for new projects? make a test suite for each project?
// to be considered.
class basicConfiguration {
public:
    static int estream() {
        string config =
"<?xml version=\"1.0\" encoding=\"UTF-8\" ?>"
"<EACIRC>"
"<INFO>"
"    <DATE>unspecified</DATE>"
"    <VERSION>5.0</VERSION>"
"    <NOTES>-</NOTES>"
"</INFO>"
"<MAIN>"
"    <PROJECT>100</PROJECT>"
"    <EVALUATOR>25</EVALUATOR>"
"    <RECOMMENCE_COMPUTATION>0</RECOMMENCE_COMPUTATION>"
"    <LOAD_INITIAL_POPULATION>0</LOAD_INITIAL_POPULATION>"
"    <NUM_GENERATIONS>20</NUM_GENERATIONS>"
"    <SAVE_STATE_FREQ>10</SAVE_STATE_FREQ>"
"</MAIN>"
"<OUTPUTS>"
"    <GRAPH_FILES>1</GRAPH_FILES>"
"    <INTERMEDIATE_CIRCUITS>0</INTERMEDIATE_CIRCUITS>"
"    <CIRCUITS_PRECISION>4</CIRCUITS_PRECISION>"
"    <ALLOW_PRUNNING>0</ALLOW_PRUNNING>"
"    <SAVE_TEST_VECTORS>0</SAVE_TEST_VECTORS>"
"</OUTPUTS>"
"<RANDOM>"
"    <USE_FIXED_SEED>0</USE_FIXED_SEED>"
"    <SEED>123456789</SEED>"
"    <BIAS_RNDGEN_FACTOR>95</BIAS_RNDGEN_FACTOR>"
"    <USE_NET_SHARE>0</USE_NET_SHARE>"
"    <QRNG_PATH>../../qrng/;C:/RNG/;D:/RandomData/</QRNG_PATH>"
"    <QRNG_MAX_INDEX>192</QRNG_MAX_INDEX>"
"</RANDOM>"
"<CUDA>"
"    <ENABLED>0</ENABLED>"
"    <SOMETHING>something</SOMETHING>"
"</CUDA>"
"<GA>"
"    <EVOLUTION_OFF>0</EVOLUTION_OFF>"
"    <PROB_MUTATION>0.05</PROB_MUTATION>"
"    <PROB_CROSSING>0.5</PROB_CROSSING>"
"    <POPULATION_SIZE>20</POPULATION_SIZE>"
"</GA>"
"<CIRCUIT>"
"    <NUM_LAYERS>5</NUM_LAYERS>"
"    <SIZE_LAYER>8</SIZE_LAYER>"
"    <SIZE_INPUT_LAYER>16</SIZE_INPUT_LAYER>"
"    <SIZE_OUTPUT_LAYER>2</SIZE_OUTPUT_LAYER>"
"    <NUM_CONNECTORS>4</NUM_CONNECTORS>"
"    <USE_MEMORY>0</USE_MEMORY>"
"    <MEMORY_SIZE>2</MEMORY_SIZE>"
"    <ALLOWED_FUNCTIONS>"
"        <FNC_NOP>1</FNC_NOP>"
"        <FNC_OR>1</FNC_OR>"
"        <FNC_AND>1</FNC_AND>"
"        <FNC_CONST>1</FNC_CONST>"
"        <FNC_XOR>1</FNC_XOR>"
"        <FNC_NOR>1</FNC_NOR>"
"        <FNC_NAND>1</FNC_NAND>"
"        <FNC_ROTL>1</FNC_ROTL>"
"        <FNC_ROTR>1</FNC_ROTR>"
"        <FNC_BITSELECTOR>1</FNC_BITSELECTOR>"
"        <FNC_SUM>1</FNC_SUM>"
"        <FNC_SUBS>1</FNC_SUBS>"
"        <FNC_ADD>1</FNC_ADD>"
"        <FNC_MULT>1</FNC_MULT>"
"        <FNC_DIV>1</FNC_DIV>"
"        <FNC_READX>1</FNC_READX>"
"        <FNC_EQUAL>0</FNC_EQUAL>"
"    </ALLOWED_FUNCTIONS>"
"</CIRCUIT>"
"<TEST_VECTORS>"
"    <INPUT_LENGTH>16</INPUT_LENGTH>"
"    <OUTPUT_LENGTH>2</OUTPUT_LENGTH>"
"    <SET_SIZE>1000</SET_SIZE>"
"    <SET_CHANGE_FREQ>5</SET_CHANGE_FREQ>"
"    <EVALUATE_BEFORE_TEST_VECTOR_CHANGE>0</EVALUATE_BEFORE_TEST_VECTOR_CHANGE>"
"    <EVALUATE_EVERY_STEP>0</EVALUATE_EVERY_STEP>"
"</TEST_VECTORS>"
"<ESTREAM>"
"    <USAGE_TYPE>101</USAGE_TYPE>"
"    <CIPHER_INIT_FREQ>1</CIPHER_INIT_FREQ>"
"    <ALGORITHM_1>10</ALGORITHM_1>"
"    <ALGORITHM_2>99</ALGORITHM_2>"
"    <BALLANCED_TEST_VECTORS>1</BALLANCED_TEST_VECTORS>"
"    <LIMIT_NUM_OF_ROUNDS>1</LIMIT_NUM_OF_ROUNDS>"
"    <ROUNDS_ALG_1>2</ROUNDS_ALG_1>"
"    <ROUNDS_ALG_2>0</ROUNDS_ALG_2>"
"    <PLAINTEXT_TYPE>0</PLAINTEXT_TYPE>"
"    <KEY_TYPE>2</KEY_TYPE>"
"    <IV_TYPE>0</IV_TYPE>"
"    <GENERATE_STREAM>0</GENERATE_STREAM>"
"    <STREAM_SIZE>1024</STREAM_SIZE>"
"</ESTREAM>"
"<SHA3>"
"    <USAGE_TYPE>201</USAGE_TYPE>"
"    <PLAINTEXT_TYPE>210</PLAINTEXT_TYPE>"
"    <USE_FIXED_SEED>0</USE_FIXED_SEED>"
"    <SEED>145091104</SEED>"
"    <ALGORITHM_1>1</ALGORITHM_1>"
"    <ALGORITHM_2>99</ALGORITHM_2>"
"    <HASHLENGTH_ALG_1>256</HASHLENGTH_ALG_1>"
"    <HASHLENGTH_ALG_2>256</HASHLENGTH_ALG_2>"
"    <BALLANCED_TEST_VECTORS>1</BALLANCED_TEST_VECTORS>"
"    <LIMIT_NUM_OF_ROUNDS>0</LIMIT_NUM_OF_ROUNDS>"
"    <ROUNDS_ALG_1>3</ROUNDS_ALG_1>"
"    <ROUNDS_ALG_2>0</ROUNDS_ALG_2>"
"    <GENERATE_STREAM>0</GENERATE_STREAM>"
"    <STREAM_SIZE>1024</STREAM_SIZE>"
"</SHA3>"
"<TEA>"
"</TEA>"
"<FILES>"
"    <USAGE_TYPE>401</USAGE_TYPE>"
"    <FILENAME_1>../../qrng/Random00.bin</FILENAME_1>"
"    <FILENAME_2>../../qrng/Random01.bin</FILENAME_2>"
"    <BALLANCED_TEST_VECTORS>1</BALLANCED_TEST_VECTORS>"
"    <USE_FIXED_INITIAL_OFFSET>1</USE_FIXED_INITIAL_OFFSET>"
"    <INITIAL_OFFSET_1>1048576</INITIAL_OFFSET_1>"
"    <INITIAL_OFFSET_2>1048000</INITIAL_OFFSET_2>"
"</FILES>"
"</EACIRC>";
        ofstream configFile(FILE_CONFIG);
        if (!configFile.is_open()) {
            return STAT_FILE_WRITE_FAIL;
        }
        configFile << config;
        configFile.close();
        return STAT_OK;
    }
};

#endif // TESTS_H
