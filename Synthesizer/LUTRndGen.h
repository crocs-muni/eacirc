#ifndef LUT_RNDGEN_H
#define LUT_RNDGEN_H

#include <string>
#include "LUT.h"
#include "RandGen.h"
#include "../core/dataset.h"
#include "../core/project.h"

class LUTRndGen : public Stream, RandGen<u64> {
public:
	LUTRndGen(unsigned long seed, int m_LevelOfRandomness = 262144);
	void read(Dataset& data);
	u64 operator()();
private:
	LUT_CTX S;
	int m_LevelOfRandomness; //number of ones () in LUT - defines level of randomness
};

#endif


