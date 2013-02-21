#ifndef ESTREAMCONSTANTS_H
#define ESTREAMCONSTANTS_H

// eStream test vector generation method
#define ESTREAM_DISTINCT            667
#define ESTREAM_PREDICT_KEY         668
#define ESTREAM_BITS_TO_CHANGE      669

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

typedef struct _ESTREAM_SETTINGS {
    int testVectorAlgorithm;
    int testVectorAlgorithm2;
    int testVectorEstreamMethod;
    int estreamKeyType;
    int estreamInputType;
    int estreamIVType;
    bool limitAlgRounds;
    int limitAlgRoundsCount;
    int limitAlgRoundsCount2;
    bool testVectorBalance;
    int cipherInitializationFrequency;
    bool generateStream;
    unsigned long streamSize;

    _ESTREAM_SETTINGS(void) {
        testVectorAlgorithm = 0;
        testVectorAlgorithm2 = 0;
        testVectorEstreamMethod = 0;
        estreamKeyType = ESTREAM_GENTYPE_ZEROS;
        estreamInputType = ESTREAM_GENTYPE_ZEROS;
        estreamIVType = ESTREAM_GENTYPE_ZEROS;
        limitAlgRounds = false;
        limitAlgRoundsCount = -1;
        limitAlgRoundsCount2 = -1;
        testVectorBalance = false;
        cipherInitializationFrequency = ESTREAM_INIT_CIPHERS_ONCE;
        generateStream = false;
        streamSize = 0;
    }
} ESTREAM_SETTINGS;

extern ESTREAM_SETTINGS* pEstreamSettings;

#endif // ESTREAMCONSTANTS_H
