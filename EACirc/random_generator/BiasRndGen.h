#include <string>
#include "IRndGen.h"
#include "RndGen.h"

#ifndef BIAS_RNDGEN_H
#define BIAS_RNDGEN_H

class BiasRndGen : public IRndGen{
	CRndGen *rndGen;
	int chanceForOne; // probability of getting bit 1 (in %)
public:
	BiasRndGen(unsigned long seed = 0, std::string QRBGSPath = "");
	BiasRndGen(unsigned long seed, string QRBGSPath, int chanceForOne);
	~BiasRndGen();
    int GetRandomFromInterval(unsigned long highBound, unsigned long *pRandom);
    int GetRandomFromInterval(unsigned char highBound, unsigned char *pRandom);
    int GetRandomFromInterval(unsigned int highBound, int *pRandom);
    int GetRandomFromInterval(float highBound, float *pRandom);
	int InitRandomGenerator(unsigned long seed = 0, std::string QRBGSPath = "");
	string ToString();
	void setChanceForOne(int chance);
    
protected:
    int UpdateAccumulator();
};

#endif
