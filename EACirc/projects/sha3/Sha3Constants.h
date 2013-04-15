#ifndef SHA3CONSTANTS_H
#define SHA3CONSTANTS_H

// SHA-3 test vector generation method
#define SHA3_DISTINGUISHER          201

// constants for test vector generation method
#define SHA3_COUNTER                210

// filenames for streams
#define SHA3_FILE_STREAM_1      "sha3_stream1.bin"
#define SHA3_FILE_STREAM_2      "sha3_stream2.bin"

// SHA-3 algorithm constants
#define SHA3_ABACUS         1
#define SHA3_ARIRANG        2
#define SHA3_AURORA         3
#define SHA3_BLAKE          4
#define SHA3_BLENDER        5
#define SHA3_BMW            6
#define SHA3_BOOLE          7
#define SHA3_CHEETAH        8
#define SHA3_CHI            9
#define SHA3_CRUNCH         10
#define SHA3_CUBEHASH       11
#define SHA3_DCH            12
#define SHA3_DYNAMICSHA     13
#define SHA3_DYNAMICSHA2    14
#define SHA3_ECHO           15
#define SHA3_ECOH           16
#define SHA3_EDON           17
#define SHA3_ENRUPT         18
#define SHA3_ESSENCE        19
#define SHA3_FUGUE          20
#define SHA3_GROSTL         21
#define SHA3_HAMSI          22
#define SHA3_JH             23
#define SHA3_KECCAK         24
#define SHA3_KHICHIDI       25
#define SHA3_LANE           26
#define SHA3_LESAMNTA       27
#define SHA3_LUFFA          28
#define SHA3_LUX            29
#define SHA3_MSCSHA3        30
#define SHA3_MD6            31
#define SHA3_MESHHASH       32
#define SHA3_NASHA          33
#define SHA3_SANDSTORM      34
#define SHA3_SARMAL         35
#define SHA3_SHABAL         36
#define SHA3_SHAMATA        37
#define SHA3_SHAVITE3       38
#define SHA3_SIMD           39
#define SHA3_SKEIN          40
#define SHA3_SPECTRALHASH   41
#define SHA3_STREAMHASH     42
#define SHA3_SWIFFTX        43
#define SHA3_TANGLE         44
#define SHA3_TIB3           45
#define SHA3_TWISTER        46
#define SHA3_VORTEX         47
#define SHA3_WAMM           48
#define SHA3_WATERFALL      49
#define SHA3_RANDOM         99

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
