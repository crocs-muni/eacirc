#include"MersenneTwisterRndGen.h"


void MersenneTwisterRndGen::read(Dataset& data) {
	u64* dataPtr = (u64*)data.data();
	int byteSize = data.num_of_tvs()*data.tv_size();
	int numIter = byteSize / 8;
	
	for (int i = 0; i < numIter; i++)
	{
		dataPtr[0] = engine();
		dataPtr++;
	}

	if ((byteSize % 8) != 0) {
		u64 tmp = engine();
		memcpy(dataPtr,&tmp, byteSize % 8);
	}
}

MersenneTwisterRndGen::MersenneTwisterRndGen(unsigned long seed) {
	engine.seed(seed);
}

u64 MersenneTwisterRndGen::operator()() {
	return engine();
}
