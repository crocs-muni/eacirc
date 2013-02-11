#include "MD5RndGen.h"
#include <random>
#include <limits.h>
#include "EACglobals.h"
#include "XMLProcessor.h"


MD5RndGen::MD5RndGen(unsigned long seed)
        : IRndGen(GENERATOR_MD5, seed) {
    if (mainGenerator == NULL) {
        // most likely this is about to become main generator
        // must create initial state deterministicly
        minstd_rand systemGenerator(m_seed);
        for (unsigned char i = 0; i < MD5_DIGEST_LENGTH; i++) {
            m_md5Accumulator[i] = systemGenerator();
        }
    } else {
        // if main generator was already initialized, use it to get new random accumulator
        for (unsigned char i = 0; i < MD5_DIGEST_LENGTH; i++) {
            mainGenerator->getRandomFromInterval((unsigned char) UCHAR_MAX, m_md5Accumulator+i);
        }
    }
}

int MD5RndGen::getRandomFromInterval(unsigned long highBound, unsigned long* pRandom) {
    int status = STAT_OK;

    // UPDATE ACCUMULATOR
    status = updateAccumulator();

    // GET FIRST DWORD FROM ACCUMULATOR     
    unsigned long random;
    memcpy(&random, m_md5Accumulator, sizeof(unsigned long));
    if (pRandom) *pRandom = (unsigned long) (((float) random / ULONG_MAX) *  highBound);

    return status;
}

int MD5RndGen::getRandomFromInterval(unsigned char highBound, unsigned char* pRandom) {
    int     status = STAT_OK;
    DWORD   rand = 0;
    
    status = getRandomFromInterval(highBound, &rand);
    *pRandom = (unsigned char) rand;

    return status;
}

int MD5RndGen::getRandomFromInterval(int highBound, int* pRandom) {
    int status = STAT_OK;

    // UPDATE ACCUMULATOR
    status = updateAccumulator();

    // GET FIRST 2 bytes FROM ACCUMULATOR
    int random;
    memcpy(&random, m_md5Accumulator, sizeof(int));
    // SUPRESS NEGATIVE VALUES
    random = abs(random);
    if (pRandom) *pRandom = (int) (((float) random / INT_MAX) *  highBound);

    return status;
}

int MD5RndGen::getRandomFromInterval(float highBound, float *pRandom) {
    int status = STAT_OK;

    // UPDATE ACCUMULATOR
    status = updateAccumulator();

    // GET FIRST 2 bytes FROM ACCUMULATOR
    unsigned long random;
    memcpy(&random, m_md5Accumulator, sizeof(unsigned long));
    if (pRandom) *pRandom = (float) (((float) random / ULONG_MAX) *  highBound);

    return status;
}

int MD5RndGen::updateAccumulator() {
    MD5_CTX mdContext;
    
    // UPDATE ACCUMULATOR
	MD5Init(&mdContext);
	MD5Update(&mdContext, m_md5Accumulator, MD5_DIGEST_LENGTH);
	MD5Final(&mdContext);
    memcpy(m_md5Accumulator, mdContext.digest, MD5_DIGEST_LENGTH);
    
    return STAT_OK;
}  

int MD5RndGen::discartValue() {
    return updateAccumulator();
}

string MD5RndGen::shortDescription() const {
    return "MD5-based generator";
}

MD5RndGen::MD5RndGen(TiXmlNode *pRoot)
    : IRndGen(GENERATOR_MD5,1) {  // cannot call IRndGen with seed 0, warning would be issued
    if (atoi(getXMLElementValue(pRoot,"@type").c_str()) != m_type) {
        mainLogger.out() << "error: Incompatible generator types." << endl;
        return;
    }

    istringstream(getXMLElementValue(pRoot,"original_seed")) >> m_seed;

    if (atol(getXMLElementValue(pRoot,"accumulator_state/@length").c_str()) != MD5_DIGEST_LENGTH) {
        mainLogger.out() << "error: Incompatible accumulator length - state not loaded." << endl;
        mainLogger.out() << "       required: " << MD5_DIGEST_LENGTH << endl;
        mainLogger.out() << "          found: " << getXMLElementValue(pRoot,"accumulator_state/@length") << endl;
    } else {
        istringstream ss(getXMLElementValue(pRoot,"accumulator_state"));
        unsigned int value;
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
            ss >> value;
            m_md5Accumulator[i] = value;
        }
    }
}

TiXmlNode* MD5RndGen::exportGenerator() const {
    TiXmlElement* pRoot = new TiXmlElement("generator");
    pRoot->SetAttribute("type",toString(m_type).c_str());
    pRoot->SetAttribute("description",shortDescription().c_str());

    TiXmlElement* originalSeed = new TiXmlElement("original_seed");
    stringstream sSeed;
    sSeed << m_seed;
    originalSeed->LinkEndChild(new TiXmlText(sSeed.str().c_str()));
    pRoot->LinkEndChild(originalSeed);

    TiXmlElement* accumulatorState = new TiXmlElement("accumulator_state");
    accumulatorState->SetAttribute("length",MD5_DIGEST_LENGTH);
    stringstream sAccValue;
    sAccValue << left << dec;
    for (unsigned char i = 0; i < MD5_DIGEST_LENGTH; i++) {
        sAccValue << (unsigned int) m_md5Accumulator[i] << " ";
    }
    accumulatorState->LinkEndChild(new TiXmlText(sAccValue.str().c_str()));
    pRoot->LinkEndChild(accumulatorState);

    return pRoot;
}
