/*********************************************************************
* Filename:   sha256.h
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Defines the API for the corresponding SHA1 implementation.
*********************************************************************/

#ifndef SHA256_H
#define SHA256_H

/*************************** HEADER FILES ***************************/
#include <stddef.h>

/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest
#define SHA256_FULL_ROUNDS 64

/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;             // 8-bit byte
typedef unsigned int  WORD;             // 32-bit word, change to "long" for 16-bit machines

typedef struct {
    BYTE data[64];
    WORD datalen;
    unsigned long long bitlen;
    WORD state[8];
} SHA256_CTX;

/*********************** FUNCTION DECLARATIONS **********************/
void sha256_init(SHA256_CTX *ctx);
void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len, unsigned rounds);
void sha256_final(SHA256_CTX *ctx, BYTE hash[], unsigned rounds);

/*********************** SHA3 adapter **********************/
#include "../../Sha3Interface.h"

class SHA256 : public Sha3Interface {

private:
    SHA256_CTX m_state;
    unsigned m_rounds;

public:
    SHA256(const int numRounds=SHA256_FULL_ROUNDS);
    int Init(int hashbitlen=256);
    int Update(const BitSequence *data, DataLength databitlen);
    int Final(BitSequence *hashval);
    int Hash(int hashbitlen, const BitSequence *data, DataLength databitlen, BitSequence *hashval);

};

#endif   // SHA256_H
