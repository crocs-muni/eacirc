#include "IRndGen.h"
#include "QuantumRndGen.h"
#include "BiasRndGen.h"
#include "MD5RndGen.h"
#include "EACglobals.h"
#include "XMLProcessor.h"

IRndGen::IRndGen(int type, unsigned long seed)
    : m_type(type) {
    // if seed not supplied externally, generate random from main generator
    if (seed == 0) {
        if (mainGenerator != NULL) {
            mainGenerator->getRandomFromInterval(ULONG_MAX,&seed);
            mainLogger.out() << "warning: Generator of type " << m_type << " initialized from Main Generator." << endl;
        } else {
            mainLogger.out() << "error: Nain generator not available, no seeding performed! (generator type " << m_type << ")" << endl;
        }
        //mainLogger.out() << "warning: Using system time to initialize random generator (type " << m_type << ")." << endl;
    }
    this->m_seed = seed;
}

IRndGen* IRndGen::parseGenerator(TiXmlNode* pRoot) {
    if (pRoot == NULL) {
        mainLogger.out() << "error: Generator could not load - NULL pointer." << endl;
        return NULL;
    }
    int generatorType = atoi(getXMLElementValue(pRoot,"@type").c_str());
    IRndGen* tempGenerator = NULL;
    switch (generatorType) {
    case GENERATOR_QRNG:
        tempGenerator = new QuantumRndGen(pRoot);
        break;
    case GENERATOR_BIAS:
        tempGenerator = new BiasRndGen(pRoot);
        break;
    case GENERATOR_MD5:
        tempGenerator = new MD5RndGen(pRoot);
        break;
    default:
        mainLogger.out() << "error: Generator could not load - unknown type (" << generatorType << ")." << endl;
        return NULL;
    }
    return tempGenerator;
}
