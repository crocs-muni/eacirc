#ifndef SHA3_ABSTRACT_H
#define SHA3_ABSTRACT_H

#include "Sha3Constants.h"

typedef unsigned char BitSequence;
typedef unsigned long long DataLength;

class Sha3Interface {

public:
    Sha3Interface() {}
    virtual ~Sha3Interface() {}
    virtual int Init(int hashbitlen) = 0;
    virtual int Update(const BitSequence *data, DataLength databitlen) = 0;
    virtual int Final(BitSequence *hashval) = 0;
    virtual int Hash(int hashbitlen, const BitSequence *data, DataLength databitlen, BitSequence *hashval) = 0;

    /** allocate new hash function object according to parameters
      * @param algorithm        hash function constant
      * @param numRounds        number of rounds used
      * @return allocated hash function obejct
      */
    static Sha3Interface* getSha3Function(int algorithm, int numRounds);

};

#endif
