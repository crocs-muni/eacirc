#include <string>
#include "IRndGen.h"

#ifndef RNDGEN_H
#define RNDGEN_H

#define		RANDOM_FILE_SIZE		10000000
#define		RANDOM_FILE_INDEX_MAX	10

class CRndGen : public IRndGen{
	unsigned char *accumulator;
	bool bQRGBS; // using rng output files?
	std::string bQRGBSPath; // path to rng output files
	int accLength; // real data length
	int accPosition; // accumulator position
	long seed; // seed
public:
	CRndGen(unsigned long seed = 0, std::string QRBGSPath = "");
    int GetRandomFromInterval(unsigned long highBound, unsigned long *pRandom);
    int GetRandomFromInterval(unsigned char highBound, unsigned char *pRandom);
    int GetRandomFromInterval(int highBound, int *pRandom);
    int GetRandomFromInterval(float highBound, float *pRandom);
	int InitRandomGenerator(unsigned long seed = 0, std::string QRBGSPath = "");
	string ToString();
    
protected:
    int UpdateAccumulator();
};

#endif