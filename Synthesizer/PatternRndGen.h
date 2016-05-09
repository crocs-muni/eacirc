#ifndef PATTERN_RNDGEN_H
#define PATTERN_RNDGEN_H

#include<vector>
#include"../core/base.h"
#include"../core/project.h"
#include "RandGen.h"
#include<random>

// polynomial is evaluated to zero - at least one random variable in each term is set to zero

class PatternRndGen : public  Stream {
public:
	PatternRndGen(std::vector<std::vector<short>>& terms, int testVectorBitSize, unsigned long seed);
	void setPolynomial();
	void setRandomData();
	void read(Dataset& data);
private:
	std::mt19937_64 gen;
    u64* _testVector;         
	int testVectorSize; 
	std::vector<std::vector<short>> _terms; //terms of the polynomial {0,1}, {2,5,7} represent polynomial = x_0x_1 + x_2x_5x_7
};
#endif //PATTERN_RNDGEN_H
