#include "IRndGen.h"
#include "QuantumRndGen.h"
#include "BiasRndGen.h"
#include "EACirc.h"

// from time, when main generator was from <random>
//minstd_rand IRndGen::m_mainGenerator;

//MD5RndGen* IRndGen::m_mainGenerator = NULL;

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

/*
void IRndGen::initMainGenerator(unsigned long seed) {
    if (seed == 0) {
        mainLogger.out() << "Seed for main generator not provided, cannot initialize." << endl;
    } else {
        m_mainGenerator = new MD5RndGen(seed);
    }
}
*/

/*
static unsigned long IRndGen::getRandomFromMainGenerator() {
    unsigned long random = 0;
    m_mainGenerator->getRandomFromInterval(ULONG_MAX, &random);
    return random;.
}
*/

/*
void IRndGen::initMainGenerator(TiXmlNode* pRoot) {
    m_mainGenerator = new MD5RndGen(pRoot);
    from time, when main generator was from <random>
    TiXmlElement* pElem = pRoot->ToElement();
    istringstream ss(pElem->GetText());
    if (strcmp(pElem->Attribute("type"),typeid(m_mainGenerator).name()) == 0) {
        ss >> m_mainGenerator;
    } else {
        mainLogger.out() << "Error: Incompatible system generator type - state not loaded." << endl;
        mainLogger.out() << "       required: " << typeid(m_mainGenerator).name() << endl;
        mainLogger.out() << "          found: " << pElem->Attribute("type") << endl;
    }

}
*/

/*
TiXmlNode* IRndGen::exportMainGenerator() {
    TiXmlElement* pRoot = new TiXmlElement("main_generator");
    pRoot->LinkEndChild(m_mainGenerator->exportGenerator());
    from time, when main generator was from <random>
    pRoot->SetAttribute("type",typeid(m_mainGenerator).name());
    stringstream state;
    state << dec << left << setfill(' ');
    state << m_mainGenerator;
    pRoot->LinkEndChild(new TiXmlText(state.str().c_str()));

    return pRoot;
}
*/
