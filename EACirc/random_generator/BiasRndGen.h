#include <string>
#include "IRndGen.h"
#include "QuantumRndGen.h"

#ifndef BIAS_RNDGEN_H
#define BIAS_RNDGEN_H

class BiasRndGen : public IRndGen{
    IRndGen* m_rndGen;
    int m_chanceForOne; // probability of getting bit 1 (in %)
public:
    BiasRndGen(unsigned long seed = 0, string QRBGSPath = "", int m_chanceForOne = 50);
    BiasRndGen(TiXmlNode* pRoot);
    // BiasRndGen(const BiasRndGen&) = delete; //(not supprrted in MS VS)
    // const BiasRndGen& operator =(const BiasRndGen&) = delete; //(not supprrted in MS VS)
    ~BiasRndGen();

    int getRandomFromInterval(unsigned long highBound, unsigned long *pRandom);
    int getRandomFromInterval(unsigned char highBound, unsigned char *pRandom);
    int getRandomFromInterval(int highBound, int *pRandom);
    int getRandomFromInterval(float highBound, float *pRandom);
    int discartValue();

	void setChanceForOne(int chance);
    string shortDescription() const;
    // implemented in XMLProcessor:
    TiXmlNode* exportGenerator() const;
protected:
    int UpdateAccumulator();
};

#endif
