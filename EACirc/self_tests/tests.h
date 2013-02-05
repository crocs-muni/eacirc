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
"    <DATE>2013-01-24</DATE>"
"    <VERSION>5.0</VERSION>"
"    <NOTES>-</NOTES>"
"</INFO>"
"<MAIN>"
"    <PROJECT>101</PROJECT>"
"    <EVALUATOR>5</EVALUATOR>"
"    <RECOMMENCE_COMPUTATION>0</RECOMMENCE_COMPUTATION>"
"    <LOAD_INITIAL_POPULATION>0</LOAD_INITIAL_POPULATION>"
"    <NUM_GENERATIONS>20</NUM_GENERATIONS>"
"    <SAVE_STATE_FREQ>10</SAVE_STATE_FREQ>"
"</MAIN>"
"<RANDOM>"
"    <USE_FIXED_SEED>0</USE_FIXED_SEED>"
"    <SEED>123456789</SEED>"
"    <BIAS_RNDGEN_FACTOR>95</BIAS_RNDGEN_FACTOR>"
"    <QRNG_PATH>../../qrng/</QRNG_PATH>"
"    <QRNG_MAX_INDEX>9</QRNG_MAX_INDEX>"
"</RANDOM>"
"<GA>"
"    <EVOLUTION_OFF>0</EVOLUTION_OFF>"
"    <PROB_MUTATION>0.05</PROB_MUTATION>"
"    <PROB_CROSSING>0.5</PROB_CROSSING>"
"    <POPULATION_SIZE>20</POPULATION_SIZE>"
"</GA>"
"<CIRCUIT>"
"    <NUM_LAYERS>5</NUM_LAYERS>"
"    <NUM_SELECTOR_LAYERS>1</NUM_SELECTOR_LAYERS>"
"    <SIZE_LAYER>8</SIZE_LAYER>"
"    <SIZE_INPUT_LAYER>16</SIZE_INPUT_LAYER>"
"    <SIZE_OUTPUT_LAYER>2</SIZE_OUTPUT_LAYER>"
"    <NUM_CONNECTORS>4</NUM_CONNECTORS>"
"    <GENOME_SIZE>6400</GENOME_SIZE>"
"    <ALLOW_PRUNNING>1</ALLOW_PRUNNING>"
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
"        <FNC_SUM>1</FNC_SUM>"
"        <FNC_SUBS>1</FNC_SUBS>"
"        <FNC_ADD>1</FNC_ADD>"
"        <FNC_MULT>1</FNC_MULT>"
"        <FNC_DIV>1</FNC_DIV>"
"        <FNC_READX>1</FNC_READX>"
"    </ALLOWED_FUNCTIONS>"
"</CIRCUIT>"
"<TEST_VECTORS>"
"    <TEST_VECTOR_LENGTH>16</TEST_VECTOR_LENGTH>"
"    <NUM_TEST_VECTORS>1000</NUM_TEST_VECTORS>"
"    <TEST_VECTOR_CHANGE_FREQ>5</TEST_VECTOR_CHANGE_FREQ>"
"    <TEST_VECTOR_CHANGE_PROGRESSIVE>0</TEST_VECTOR_CHANGE_PROGRESSIVE>"
"    <SAVE_TEST_VECTORS>1</SAVE_TEST_VECTORS>"
"    <EVALUATE_BEFORE_TEST_VECTOR_CHANGE>1</EVALUATE_BEFORE_TEST_VECTOR_CHANGE>"
"    <EVALUATE_EVERY_STEP>0</EVALUATE_EVERY_STEP>"
"</TEST_VECTORS>"
"<ESTREAM>"
"    <ESTREAM_USAGE_TYPE>667</ESTREAM_USAGE_TYPE>"
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
"</ESTREAM>"
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
