#include "IRndGen.h"
#include "QuantumRndGen.h"
#include "BiasRndGen.h"
#include "EACirc.h"

minstd_rand IRndGen::m_mainGenerator;

void IRndGen::initMainGenerator(TiXmlNode* pRoot) {
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

TiXmlNode* IRndGen::exportMainGenerator() {
    TiXmlElement* pRoot = new TiXmlElement("main_generator");
    pRoot->SetAttribute("type",typeid(m_mainGenerator).name());
    stringstream state;
    state << dec << left << setfill(' ');
    state << m_mainGenerator;
    pRoot->LinkEndChild(new TiXmlText(state.str().c_str()));
    return pRoot;
}
