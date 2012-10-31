//#include "stdafx.h"
#include "BiasRndGen.h"
//#include "Time.h"
#include "EACirc.h"
#include "random_generator/QuantumRndGen.h"

BiasRndGen::BiasRndGen(unsigned long seed, string QRBGSPath, int chanceForOne) {
    m_type = GENERATOR_BIAS;
    m_chanceForOne = chanceForOne;
    m_rndGen = new QuantumRndGen(seed, QRBGSPath);
}

BiasRndGen::~BiasRndGen() {
    delete this->m_rndGen;
}

int BiasRndGen::getRandomFromInterval(unsigned long highBound, unsigned long *pRandom) {
    int status = STAT_OK;
    int val;
	unsigned long random = 0;
	if (pRandom) *pRandom = 0;
	else return status;

	for (int i=0; i<32; i++){
		//NO NEED TO CONTINUE, WE CAN RETURN DIRECTLY FIRST I VALUES
		if ((highBound+1) == pGACirc->precompPow[i]){
			*pRandom = random;
			break;
		}

        m_rndGen->getRandomFromInterval(100,&val);
		if (val < m_chanceForOne)
			random|=pGACirc->precompPow[i];
	}

	if (*pRandom == 0) {
		*pRandom = (unsigned long) (((float) random / ULONG_MAX) *  highBound);
	}

	// UPDATE ACCUMULATOR
    //UpdateAccumulator();

    return status;
}

int BiasRndGen::getRandomFromInterval(unsigned char highBound, unsigned char *pRandom) {
    int status = STAT_OK;
    unsigned long   rand = 0;
    
    status = getRandomFromInterval(highBound, &rand);
    *pRandom = (unsigned char) rand;

    return status;
}

int BiasRndGen::getRandomFromInterval(int highBound, int *pRandom) {
    int status = STAT_OK;
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

        m_rndGen->getRandomFromInterval(100,&val);
		if (val < m_chanceForOne)
			random|=pGACirc->precompPow[i];
	}

	if (*pRandom == 0) {
		// SUPRESS NEGATIVE VALUES
		random = abs(random);
		*pRandom = (int) (((float) random / INT_MAX) *  highBound);
	}

	// UPDATE ACCUMULATOR
    //UpdateAccumulator();

    return status;
}

int BiasRndGen::getRandomFromInterval(float highBound, float *pRandom) {
    int status = STAT_OK;
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

        m_rndGen->getRandomFromInterval(100,&val);
		if (val < m_chanceForOne)
			random|=pGACirc->precompPow[i];
	}

	if (*pRandom == 0)
		*pRandom = (float) (((float) random / ULONG_MAX) *  highBound);

	// UPDATE ACCUMULATOR
    //UpdateAccumulator();

    return status;
}

int BiasRndGen::discartValue() {
    return m_rndGen->discartValue();
}

int BiasRndGen::reinitRandomGenerator() {
    return m_rndGen->reinitRandomGenerator();
}

/*
int BiasRndGen::UpdateAccumulator() {
    return STAT_OK;
}
*/

void BiasRndGen::setChanceForOne(int chance) {
	m_chanceForOne = chance;
}

string BiasRndGen::shortDescription() const {
    string mes = "BIAS GENERATOR ";
	stringstream out;
	out << m_chanceForOne;
	mes+=out.str();
	mes+="%";
	return mes;
}
