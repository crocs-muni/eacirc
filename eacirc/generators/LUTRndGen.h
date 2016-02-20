#include <string>
#include "IRndGen.h"
#include "QuantumRndGen.h"
#include "LUT.h"

#ifndef LUT_RNDGEN_H
#define LUT_RNDGEN_H

class LUTRndGen : public IRndGen{
	LUT_CTX S;
	int m_LevelOfRandomness; //number of ones () in LUT - defines level of randomness 
public:
	LUTRndGen(unsigned long seed = 0, int m_LevelOfRandomness = 262144); //best random generator for half(262144) of ones on LUTs 
	LUTRndGen(TiXmlNode* pRoot);
	~LUTRndGen();

	int getRandomFromInterval(unsigned long highBound, unsigned long *pRandom);
	int getRandomFromInterval(unsigned char highBound, unsigned char *pRandom);
	int getRandomFromInterval(unsigned int highBound, unsigned int *pRandom);
	int getRandomFromInterval(int highBound, int *pRandom);
	int getRandomFromInterval(float highBound, float *pRandom);
	int discartValue();

	void setLevelOfRandomness(int HW);
	string shortDescription() const;
	// implemented in XMLProcessor:
	TiXmlNode* exportGenerator() const;
protected:
	int UpdateAccumulator();
};



#endif
