//#include "stdafx.h"
#include "BiasRndGen.h"
//#include "Time.h"
#include "EACirc.h"

BiasRndGen::BiasRndGen(unsigned long seed, string QRBGSPath) {
	BiasRndGen(seed,QRBGSPath,50);
}

BiasRndGen::~BiasRndGen() {
	delete this->rndGen;
}

BiasRndGen::BiasRndGen(unsigned long seed, string QRBGSPath, int chanceForOne) {
    InitRandomGenerator(seed, QRBGSPath);
	this->chanceForOne = chanceForOne;
}


int BiasRndGen::GetRandomFromInterval(unsigned long highBound, unsigned long *pRandom) {
    int     status = STAT_OK;
	int		val;
	unsigned long random = 0;
	if (pRandom) *pRandom = 0;
	else return status;

	for (int i=0; i<32; i++){
		//NO NEED TO CONTINUE, WE CAN RETURN DIRECTLY FIRST I VALUES
		if ((highBound+1) == pGACirc->precompPow[i]){
			*pRandom = random;
			break;
		}

		rndGen->GetRandomFromInterval(100,&val);
		if (val < chanceForOne)
			random|=pGACirc->precompPow[i];
	}

	if (*pRandom == 0) {
		*pRandom = (unsigned long) (((float) random / ULONG_MAX) *  highBound);
	}

	// UPDATE ACCUMULATOR
    UpdateAccumulator();

    return status;
}

int BiasRndGen::GetRandomFromInterval(unsigned char highBound, unsigned char *pRandom) {
    int     status = STAT_OK;
    unsigned long   rand = 0;
    
    status = GetRandomFromInterval(highBound, &rand);
    *pRandom = (unsigned char) rand;

    return status;
}

int BiasRndGen::GetRandomFromInterval(unsigned int highBound, int *pRandom) {
    int     status = STAT_OK;
	int val;
    int random = 0;
	if (pRandom) *pRandom = 0;
	else return status;

	for (int i=0; i<31; i++){
		//NO NEED TO CONTINUE, WE CAN RETURN DIRECTLY FIRST I VALUES OF RANDOM
		if ((highBound+1) == pGACirc->precompPow[i]){
			*pRandom = random;
			break;
		}

		rndGen->GetRandomFromInterval(100,&val);
		if (val < chanceForOne)
			random|=pGACirc->precompPow[i];
	}

	if (*pRandom == 0) {
		// SUPRESS NEGATIVE VALUES
		random = abs(random);
		*pRandom = (int) (((float) random / INT_MAX) *  highBound);
	}

	// UPDATE ACCUMULATOR
    UpdateAccumulator();

    return status;
}

int BiasRndGen::GetRandomFromInterval(float highBound, float *pRandom) {
    int     status = STAT_OK;
	int val;
    unsigned long random = 0;
	if (pRandom) *pRandom = 0;
	else return status;


	for (int i=0; i<32; i++) {
		//NO NEED TO CONTINUE, WE CAN RETURN DIRECTLY FIRST I VALUES OF RANDOM
		if ((highBound+1) == pGACirc->precompPow[i]){
			*pRandom = random;
			break;
		}

		rndGen->GetRandomFromInterval(100,&val);
		if (val < chanceForOne)
			random|=pGACirc->precompPow[i];
	}

	if (*pRandom == 0)
		*pRandom = (float) (((float) random / ULONG_MAX) *  highBound);

	// UPDATE ACCUMULATOR
    UpdateAccumulator();

    return status;
}

int BiasRndGen::InitRandomGenerator(unsigned long seed, string QRBGSPath) {
	this->rndGen = new CRndGen(seed, QRBGSPath);
	return STAT_OK;
}

int BiasRndGen::UpdateAccumulator() {
    return STAT_OK;
}  

void BiasRndGen::setChanceForOne(int chance) {
	chanceForOne = chance;
}

string BiasRndGen::ToString() {
	string mes = "BIAS GENERATOR ";	
	stringstream out;
	out << chanceForOne;
	mes+=out.str();
	mes+="%";
	return mes;
}
