#ifndef STREAMHASH_SHA3_H
#define STREAMHASH_SHA3_H

#include "../../Sha3Interface.h"

class StreamHash : public SHA3 {

typedef enum { SUCCESS=0, FAIL=1, BAD_HASHLEN=2 } HashReturn;

typedef struct {
    int hashbitlen, bitlen, tablen;
    unsigned int tabval[16];
    BitSequence bitval; /* unprocessed bits */
} hashState;

private:
hashState streamhashState;

public:
int Init(int hashbitlen);
int Update(const BitSequence *data,
        DataLength databitlen);
int Final(BitSequence *hashval);
int Hash(int hashbitlen, const BitSequence *data,
        DataLength databitlen, BitSequence *hashval);

};

#endif