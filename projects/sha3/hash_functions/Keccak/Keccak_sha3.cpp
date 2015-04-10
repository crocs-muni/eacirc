#include <string.h>
#include "Keccak_sha3.h"
extern "C" {
#include "KeccakF-1600-interface.h"
}

int Keccak::Init(int hashbitlen)
{
    switch(hashbitlen) {
        case 0: // Default parameters, arbitrary length output
            InitSponge((spongeState*)&keccakState, 1024, 576);
            break;
        case 224:
            InitSponge((spongeState*)&keccakState, 1152, 448);
            break;
        case 256:
            InitSponge((spongeState*)&keccakState, 1088, 512);
            break;
        case 384:
            InitSponge((spongeState*)&keccakState, 832, 768);
            break;
        case 512:
            InitSponge((spongeState*)&keccakState, 576, 1024);
            break;
        default:
            return BAD_HASHLEN;
    }
    keccakState.fixedOutputLength = hashbitlen;
    return SUCCESS;
}

int Keccak::Update(const BitSequence *data, DataLength databitlen)
{
    if ((databitlen % 8) == 0)
        return Absorb((spongeState*)&keccakState, data, databitlen);
    else {
        int ret = Absorb((spongeState*)&keccakState, data, databitlen - (databitlen % 8));
        if (ret == SUCCESS) {
            unsigned char lastByte; 
            // Align the last partial byte to the least significant bits
            lastByte = data[databitlen/8] >> (8 - (databitlen % 8));
            return Absorb((spongeState*)&keccakState, &lastByte, databitlen % 8);
        }
        else
            return ret;
    }
}

int Keccak::Final(BitSequence *hashval)
{
    return Squeeze(&keccakState, hashval, keccakState.fixedOutputLength);
}

int Keccak::Hash(int hashbitlen, const BitSequence *data, DataLength databitlen, BitSequence *hashval)
{
    //hashState state;
    //HashReturn result;
	int result;

    if ((hashbitlen != 224) && (hashbitlen != 256) && (hashbitlen != 384) && (hashbitlen != 512))
        return BAD_HASHLEN; // Only the four fixed output lengths available through this API
    result = Keccak::Init(hashbitlen);
    if (result != SUCCESS)
        return result;
    result = Keccak::Update(data, databitlen);
    if (result != SUCCESS)
        return result;
    result = Keccak::Final(hashval);
    return result;
}