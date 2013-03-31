#ifndef ESTREAMCONSTANTS_H
#define ESTREAMCONSTANTS_H

// eStream test vector generation method
#define ESTREAM_DISTINCT            101
#define ESTREAM_PREDICT_KEY         102
#define ESTREAM_BITS_TO_CHANGE      103

// project-specific evaluators
#define ESTREAM_EVALUATOR_AVALANCHE 126

// constants for cipher initialization frequency
#define ESTREAM_INIT_CIPHERS_ONCE           0
#define ESTREAM_INIT_CIPHERS_FOR_SET        1
#define ESTREAM_INIT_CIPHERS_FOR_VECTOR     2

// eStream data types (for key, iv, plaintext)
#define ESTREAM_GENTYPE_ZEROS       0
#define ESTREAM_GENTYPE_ONES        1
#define ESTREAM_GENTYPE_RANDOM      2
#define ESTREAM_GENTYPE_BIASRANDOM  3

// cipher constants
#define STREAM_BLOCK_SIZE       16

// eStream cipher constants
#define ESTREAM_ABC             1
#define ESTREAM_ACHTERBAHN      2
#define ESTREAM_CRYPTMT         3
#define ESTREAM_DECIM           4
#define ESTREAM_DICING          5
#define ESTREAM_DRAGON          6
#define ESTREAM_EDON80          7
#define ESTREAM_FFCSR           8
#define ESTREAM_FUBUKI          9
#define ESTREAM_GRAIN           10
#define ESTREAM_HC128           11
#define ESTREAM_HERMES          12
#define ESTREAM_LEX             13
#define ESTREAM_MAG             14
#define ESTREAM_MICKEY          15
#define ESTREAM_MIR1            16
#define ESTREAM_POMARANCH       17
#define ESTREAM_PY              18
#define ESTREAM_RABBIT          19
#define ESTREAM_SALSA20         20
#define ESTREAM_SFINKS          21
#define ESTREAM_SOSEMANUK       22
#define ESTREAM_TRIVIUM         23
#define ESTREAM_TSC4            24
#define ESTREAM_WG              25
#define ESTREAM_YAMB            26
#define ESTREAM_ZKCRYPT         27
#define ESTREAM_RANDOM          99

struct ESTREAM_SETTINGS {
    int estreamUsageType;
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
        estreamUsageType = 0;
        cipherInitializationFrequency = ESTREAM_INIT_CIPHERS_ONCE;
        algorithm1 = 0;
        algorithm2 = 0;
        ballancedTestVectors = false;
        limitAlgRounds = false;
        alg1RoundsCount = -1;
        alg2RoundsCount = -1;
        keyType = ESTREAM_GENTYPE_ZEROS;
        plaintextType = ESTREAM_GENTYPE_ZEROS;
        ivType = ESTREAM_GENTYPE_ZEROS;
        generateStream = false;
        streamSize = 0;
    }
};

extern ESTREAM_SETTINGS* pEstreamSettings;

#endif // ESTREAMCONSTANTS_H
