#ifndef SHA3CONSTANTS_H
#define SHA3CONSTANTS_H

// SHA-3 test vector generation method
#define SHA3_DISTINGUISHER          201

// constants for test vector generation method
#define SHA3_COUNTER                210

// filenames for streams
#define SHA3_FILE_STREAM_1      "sha3_stream1.bin"
#define SHA3_FILE_STREAM_2      "sha3_stream2.bin"

struct SHA3_SETTINGS {
    int usageType;
    int plaintextType;
    bool useFixedSeed;
    unsigned long seed;
    int algorithm1;
    int algorithm2;
    int hashLength1;
    int hashLength2;
    bool ballancedTestVectors;
    bool limitAlgRounds;
    int alg1RoundsCount;
    int alg2RoundsCount;
    bool generateStream;
    unsigned long streamSize;

    SHA3_SETTINGS(void) {
        usageType = -1;
        plaintextType = -1;
        useFixedSeed = false;
        seed = 0;
        algorithm1 = -1;
        algorithm2 = -1;
        hashLength1 = -1;
        hashLength2 = -1;
        ballancedTestVectors = false;
        limitAlgRounds = false;
        alg1RoundsCount = -1;
        alg2RoundsCount = -1;
        generateStream = false;
        streamSize = 0;
    }
};

extern SHA3_SETTINGS* pSha3Settings;

#endif // SHA3CONSTANTS_H
