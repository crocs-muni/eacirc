#ifndef KECCAK_SHA3_H
#define KECCAK_SHA3_H

#include "../../Sha3Interface.h"
extern "C" {
#include "KeccakSponge.h"
}

class Keccak : public Sha3Interface {

/*typedef unsigned char BitSequence;
typedef unsigned long long DataLength;*/
typedef enum { SUCCESS = 0, FAIL = 1, BAD_HASHLEN = 2 } HashReturn;

typedef spongeState hashState;

private:
hashState keccakState;

public:
int Init(int hashbitlen);
int Update(const BitSequence *data, DataLength databitlen);
int Final(BitSequence *hashval);
int Hash(int hashbitlen, const BitSequence *data, DataLength databitlen, BitSequence *hashval);

};

#endif