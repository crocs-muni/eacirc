#ifndef CAESARCONSTANTS_H
#define CAESARCONSTANTS_H

// CAESAR test vector generation method
#define CAESAR_DISTINGUISHER          301

// CAESAR algorithm constants
#define CAESAR_AES128CGM        1

typedef unsigned char bits_t;
typedef unsigned long long length_t;

// TODO unify message/plaintext naming

struct CAESAR_SETTINGS {
    int usageType;
    bool useFixedSeed;
    unsigned long seed;
    int algorithm;
    bool limitAlgRounds;
    int algorithmRoundsCount;
    length_t plaintextLength;
    length_t adLength;
    int plaintextType;
    int keyType;
    int adType;
    int smnType;
    int pmnType;
    bool generateStream;
    unsigned long streamSize;
    // automatically set values
    length_t ciphertextLength;

    CAESAR_SETTINGS(void) {
        usageType = -1;
        useFixedSeed = false;
        seed = 0;
        algorithm = -1;
        limitAlgRounds = false;
        algorithmRoundsCount = -1;
        plaintextLength = -1;
        adLength = -1;
        plaintextType = -1;
        keyType = -1;
        adType = -1;
        smnType = -1;
        pmnType = -1;
        generateStream = false;
        streamSize = 0;
        ciphertextLength = 0;
    }
};

extern CAESAR_SETTINGS* pCaesarSettings;

#endif // CAESARCONSTANTS_H
