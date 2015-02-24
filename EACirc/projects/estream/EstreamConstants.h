#ifndef ESTREAMCONSTANTS_H
#define ESTREAMCONSTANTS_H

// eStream usage type
#define ESTREAM_DISTINGUISHER       101
#define ESTREAM_PREDICT_KEY         102
#define ESTREAM_BITS_TO_CHANGE      103

// project-specific evaluators
#define ESTREAM_EVALUATOR_AVALANCHE 126

// constants for cipher initialization frequency
#define ESTREAM_INIT_CIPHERS_ONCE           0
#define ESTREAM_INIT_CIPHERS_FOR_SET        1
#define ESTREAM_INIT_CIPHERS_FOR_VECTOR     2

// eStream data types (for key, iv, plaintext)
#define ESTREAM_GENTYPE_ZEROS        0
#define ESTREAM_GENTYPE_ONES         1
#define ESTREAM_GENTYPE_RANDOM       2
#define ESTREAM_GENTYPE_BIASRANDOM   3
#define ESTREAM_GENTYPE_COUNTER      4
#define ESTREAM_GENTYPE_FLIP5BITS    5
#define ESTREAM_GENTYPE_HALFBLOCKSAC 6

// filenames for streams
#define ESTREAM_FILE_STREAM_1      "estream_stream1.bin"
#define ESTREAM_FILE_STREAM_2      "estream_stream2.bin"

// cipher constants
#define STREAM_BLOCK_SIZE       16

#include "EstreamCiphers.h"

struct ESTREAM_SETTINGS {
    int usageType;
    int cipherInitializationFrequency;
    int algorithm1;
    int algorithm2;
    bool ballancedTestVectors;
    bool limitAlgRounds;
    int alg1RoundsCount;
    int alg2RoundsCount;
    int keyType;
    int plaintextType;
    int ivType;
    bool generateStream;
    unsigned long streamSize;

    ESTREAM_SETTINGS(void) {
        usageType = -1;
        cipherInitializationFrequency = -1;
        algorithm1 = -1;
        algorithm2 = -1;
        ballancedTestVectors = false;
        limitAlgRounds = false;
        alg1RoundsCount = -1;
        alg2RoundsCount = -1;
        keyType = -1;
        plaintextType = -1;
        ivType = -1;
        generateStream = false;
        streamSize = -1;
    }
};

extern ESTREAM_SETTINGS* pEstreamSettings;

#endif // ESTREAMCONSTANTS_H
