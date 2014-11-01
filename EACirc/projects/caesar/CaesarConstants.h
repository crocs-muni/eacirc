#ifndef CAESARCONSTANTS_H
#define CAESARCONSTANTS_H

// CAESAR test vector generation method
#define CAESAR_DISTINGUISHER          301

// CAESAR algorithm constants
#define CAESAR_AES128CGM        1

typedef unsigned char bits_t;
typedef unsigned long long length_t;

struct CAESAR_SETTINGS {
    int usageType;
    bool useFixedSeed;
    unsigned long seed;
    int algorithm;
    bool limitAlgRounds;
    int algorithmRoundsCount;
    int plaintextType;
    int keyType;
    int ivType;
    bool generateStream;
    unsigned long streamSize;

    CAESAR_SETTINGS(void) {
        usageType = -1;
        useFixedSeed = false;
        seed = 0;
        algorithm = -1;
        limitAlgRounds = false;
        algorithmRoundsCount = -1;
        plaintextType = -1;
        keyType = -1;
        ivType = -1;
        generateStream = false;
        streamSize = 0;
    }
};

extern CAESAR_SETTINGS* pCaesarSettings;

#endif // CAESARCONSTANTS_H
