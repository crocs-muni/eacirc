#include "MD5RndGen.h"

MD5RndGen::MD5RndGen(unsigned long seed)
        : IRndGen(GENERATOR_MD5, seed) {
    reinitRandomGenerator();
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

int MD5RndGen::reinitRandomGenerator() {
    // INITIALIZE STARTUP MD5 ACUMULLATOR
    for (unsigned char i = 0; i < MD5_DIGEST_LENGTH; i++) {
        unsigned char value;
        if (mainGenerator == NULL) {
            getRandomFromInterval((unsigned char) UCHAR_MAX, &value);
        } else {
            mainGenerator->getRandomFromInterval((unsigned char) UCHAR_MAX, &value);
        }
        m_md5Accumulator[i] = value;
    }
    return STAT_OK;
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

MD5RndGen::MD5RndGen(TiXmlNode* pRoot)
    : IRndGen(GENERATOR_MD5,1) {  // cannot call IRndGen with seed 0, warning would be issued

    TiXmlElement* pElem = NULL;

    pElem = pRoot->FirstChildElement("original_seed");
    m_seed = atol(pElem->GetText());

    pElem = pRoot->FirstChildElement("accumulator_state");
    if (atol(pElem->Attribute("length")) != MD5_DIGEST_LENGTH) {
        mainLogger.out() << "Error: Incompatible accumulator length - state not loaded." << endl;
        mainLogger.out() << "       required: " << MD5_DIGEST_LENGTH << endl;
        mainLogger.out() << "          found: " << pElem->Attribute("length") << endl;
    } else {
        istringstream ss(pElem->GetText());
        unsigned char value;
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
            ss >> value;
            m_md5Accumulator[i] = value;
        }
    }
}

TiXmlNode* MD5RndGen::exportGenerator() const {
    TiXmlElement* pRoot = new TiXmlElement("generator");
    pRoot->SetAttribute("type",shortDescription().c_str());

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
