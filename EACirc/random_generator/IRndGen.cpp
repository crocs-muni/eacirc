#include "IRndGen.h"
#include "QuantumRndGen.h"
#include "BiasRndGen.h"
#include "MD5RndGen.h"
#include "EACirc.h"

IRndGen::IRndGen(int type, unsigned long seed)
    : m_type(type) {
    // if seed not supplied externally, generate random from main generator
    if (seed == 0) {
        if (mainGenerator != NULL) {
            mainGenerator->getRandomFromInterval(ULONG_MAX,&seed);
            mainLogger.out() << "Warning: generator of type " << m_type << " initialized from Main Generator." << endl;
        } else {
            mainLogger.out() << "Error: main generator not available, no seeding performed! (generator type " << m_type << ")" << endl;
        }
        //mainLogger.out() << "Warning: using system time to initialize random generator (type " << m_type << ")." << endl;
    }
    this->m_seed = seed;
}

IRndGen* IRndGen::parseGenerator(TiXmlElement* pRoot) {
    const char* typeChar = pRoot->Attribute("type");
    if (typeChar == NULL) return NULL;
    int generatorType = atoi(typeChar);
    IRndGen* tempGenerator = NULL;
    switch (generatorType) {
    case GENERATOR_QRNG: {
            tempGenerator = new QuantumRndGen(pRoot);
            break;
    }
    case GENERATOR_BIAS: {
        tempGenerator = new BiasRndGen(pRoot);
        break;
    }
    case GENERATOR_MD5: {
        tempGenerator = new MD5RndGen(pRoot);
        break;
    }
    default: {
        mainLogger.out() << "generator load error: unknown type (" << generatorType << ")." << endl;
        return NULL;
    }
    }
    // TODO: check sanity bit
    if (true) {
        return tempGenerator;
    } else {
        return NULL;
    }
}
