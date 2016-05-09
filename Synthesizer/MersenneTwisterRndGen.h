#ifndef MERSENNETWISTER_RNDGEN_H
#define MERSENNETWISTER_RNDGEN_H

#include<vector>
#include"../core/base.h"
#include"../core/project.h"
#include "RandGen.h"
#include<random>

// polynomial is evaluated to zero - at least one random variable in each term is set to zero

class MersenneTwisterRndGen : public  Stream, RandGen<u64> {
public:
	MersenneTwisterRndGen( unsigned long seed);
	void read(Dataset& data);
	u64 operator()();
private:
	std::mt19937_64  engine;
};
#endif //MERSENNETWISTER_RNDGEN_H
