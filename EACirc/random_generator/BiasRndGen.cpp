#include "BiasRndGen.h"
#include "EACglobals.h"
#include "XMLProcessor.h"
#include "random_generator/QuantumRndGen.h"

BiasRndGen::BiasRndGen(unsigned long seed, string QRBGSPath, int chanceForOne)
        : IRndGen(GENERATOR_BIAS, seed), m_chanceForOne(chanceForOne) {
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
		if ((highBound+1) == pGlobals->precompPow[i]){
			*pRandom = random;
			break;
		}

        m_rndGen->getRandomFromInterval(100,&val);
		if (val < m_chanceForOne)
			random|=pGlobals->precompPow[i];
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
        if ((unsigned long) (highBound+1) == pGlobals->precompPow[i]){
			*pRandom = random;
			break;
		}

        m_rndGen->getRandomFromInterval(100,&val);
		if (val < m_chanceForOne)
			random|=pGlobals->precompPow[i];
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
		if ((highBound+1) == pGlobals->precompPow[i]){
			*pRandom = random;
			break;
		}

        m_rndGen->getRandomFromInterval(100,&val);
		if (val < m_chanceForOne)
			random|=pGlobals->precompPow[i];
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

/*
int BiasRndGen::UpdateAccumulator() {
    return STAT_OK;
}
*/

void BiasRndGen::setChanceForOne(int chance) {
	m_chanceForOne = chance;
}

string BiasRndGen::shortDescription() const {
    string mes = "bias generetor ";
	stringstream out;
	out << m_chanceForOne;
	mes+=out.str();
	mes+="%";
	return mes;
}

BiasRndGen::BiasRndGen(TiXmlNode* pRoot)
        : IRndGen(GENERATOR_BIAS,1) {  // cannot call IRndGen with seed 0, warning would be issued
    if (atoi(getXMLElementValue(pRoot,"@type").c_str()) != m_type) {
        mainLogger.out(LOGGER_ERROR) << "incompatible generator types." << endl;
        return;
    }

    m_chanceForOne = atoi(getXMLElementValue(pRoot,"chance_for_one").c_str());
    m_rndGen = new QuantumRndGen(getXMLElement(pRoot,"internal_rng/generator"));
}

TiXmlNode* BiasRndGen::exportGenerator() const {
    TiXmlElement* pRoot = new TiXmlElement("generator");
    pRoot->SetAttribute("type",toString(m_type).c_str());
    pRoot->SetAttribute("description",shortDescription().c_str());

    TiXmlElement* chanceForOne = new TiXmlElement("chance_for_one");
    stringstream sChanceForOne;
    sChanceForOne << m_chanceForOne;
    chanceForOne->LinkEndChild(new TiXmlText(sChanceForOne.str().c_str()));
    pRoot->LinkEndChild(chanceForOne);

    pRoot->LinkEndChild(new TiXmlComment("follows state of internal QRNG"));
    TiXmlElement* pElem = new TiXmlElement("internal_rng");
    pElem->LinkEndChild(m_rndGen->exportGenerator());
    pRoot->LinkEndChild(pElem);

    return pRoot;
}
