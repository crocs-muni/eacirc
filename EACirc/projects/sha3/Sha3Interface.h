#ifndef SHA3_ABSTRACT_H
#define SHA3_ABSTRACT_H

typedef unsigned char BitSequence;
typedef unsigned long long DataLength;

class SHA3 {

public:
	virtual int Init(int hashbitlen) = 0;
	virtual int Update(const BitSequence *data, DataLength databitlen) = 0;
	virtual int Final(BitSequence *hashval) = 0;
	virtual int Hash(int hashbitlen, const BitSequence *data, DataLength databitlen, BitSequence *hashval) = 0;
};

#endif