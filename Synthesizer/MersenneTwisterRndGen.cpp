#include"MersenneTwisterRndGen.h"


void MersenneTwisterRndGen::read(Dataset& data) {
	u64* dataPtr = (u64*)data.data();
	int numIter = data.num_of_tvs()*data.tv_size() / 8;
	
	for (int i = 0; i < numIter; i++)
	{
		dataPtr[i] = engine();
	}
}

MersenneTwisterRndGen::MersenneTwisterRndGen(unsigned long seed) {
	engine.seed(seed);
}

u64 MersenneTwisterRndGen::operator()() {
	return engine();
}
