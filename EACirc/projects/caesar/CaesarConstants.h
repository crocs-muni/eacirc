#ifndef CAESARCONSTANTS_H
#define CAESARCONSTANTS_H

// CAESAR test vector generation method
#define CAESAR_DISTINGUISHER          301

// CAESAR data types (for key, plaintext, ad, smn, pmn)
#define CAESAR_TYPE_ZEROS       0
#define CAESAR_TYPE_RANDOM      1
#define CAESAR_TYPE_COUNTER     2

// CAESAR algorithm constants
#define CAESAR_AESGCM           1
#define CAESAR_ACORN            2
#define CAESAR_OCB              3

typedef unsigned char bits_t;
typedef unsigned long long length_t;

struct CAESAR_SETTINGS {
    int usageType;
    int algorithm;
    int algorithmMode;
    bool limitAlgRounds;
    int algorithmRoundsCount;
    length_t plaintextLength;
    length_t adLength;          // associated data
    int plaintextType;
    int keyType;
    int adType;                 // associated data
    int smnType;
    int pmnType;
    bool generateStream;
    unsigned long streamSize;
    // automatically set values
    length_t keyLength;
    length_t cipertextOverhead; // maximum overhead of ciphertext to plaintext
    length_t ciphertextLength;  // maximum ciphertext length for set plaintext length
    length_t smnLength;         // secret message number length
    length_t pmnLength;         // public message number length

    CAESAR_SETTINGS(void) {
        usageType = -1;
        algorithm = -1;
        algorithmMode = -1;
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
        keyLength = 0;
        cipertextOverhead = 0;
        ciphertextLength = 0;
        smnLength = 0;
        pmnLength = 0;
    }
};

extern CAESAR_SETTINGS* pCaesarSettings;

#endif // CAESARCONSTANTS_H
