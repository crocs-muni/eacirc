#include "CaesarProject.h"

CAESAR_SETTINGS* pCaesarSettings = NULL;

CaesarProject::CaesarProject()
    : IProject(PROJECT_CAESAR), m_encryptor(NULL) {}

CaesarProject::~CaesarProject() {
    if (m_encryptor != NULL) delete m_encryptor;
    m_encryptor = NULL;
}

string CaesarProject::shortDescription() const {
    return "CAESAR -- competition for authenticated encryption";
}

string CaesarProject::testingConfiguration() {
    string config =
            "<CAESAR>"
            "    <USAGE_TYPE>301</USAGE_TYPE>"
            "    <USE_FIXED_SEED>0</USE_FIXED_SEED>"
            "    <SEED>145091104</SEED>"
            "    <ALGORITHM>1</ALGORITHM>"
            "    <LIMIT_NUM_OF_ROUNDS>0</LIMIT_NUM_OF_ROUNDS>"
            "    <ALGORITHM_ROUNDS>3</ALGORITHM_ROUNDS>"
            "    <PLAINTEXT_TYPE>0</PLAINTEXT_TYPE>"
            "    <KEY_TYPE>2</KEY_TYPE>"
            "    <IV_TYPE>0</IV_TYPE>"
            "    <GENERATE_STREAM>0</GENERATE_STREAM>"
            "    <STREAM_SIZE>5242880</STREAM_SIZE>"
            "</CAESAR>";
    return config;
}

int CaesarProject::loadProjectConfiguration(TiXmlNode* pRoot) {
    m_caesarSettings.usageType = atoi(getXMLElementValue(pRoot,"CAESAR/USAGE_TYPE").c_str());
    m_caesarSettings.useFixedSeed = (atoi(getXMLElementValue(pRoot,"CAESAR/USE_FIXED_SEED").c_str())) ? true : false;
    istringstream(getXMLElementValue(pRoot,"CAESAR/SEED")) >> m_caesarSettings.seed;
    m_caesarSettings.algorithm = atoi(getXMLElementValue(pRoot,"CAESAR/ALGORITHM").c_str());
    m_caesarSettings.limitAlgRounds = (atoi(getXMLElementValue(pRoot,"CAESAR/LIMIT_NUM_OF_ROUNDS").c_str())) ? true : false;
    m_caesarSettings.algorithmRoundsCount = atoi(getXMLElementValue(pRoot,"CAESAR/ROUNDS_ALGORITHM").c_str());
    m_caesarSettings.plaintextType = atoi(getXMLElementValue(pRoot,"CAESAR/PLAINTEXT_TYPE").c_str());
    m_caesarSettings.keyType = atoi(getXMLElementValue(pRoot,"CAESAR/KEY_TYPE").c_str());
    m_caesarSettings.ivType = atoi(getXMLElementValue(pRoot,"CAESAR/IV_TYPE").c_str());
    m_caesarSettings.generateStream = (atoi(getXMLElementValue(pRoot,"CAESAR/GENERATE_STREAM").c_str())) ? true : false;
    istringstream(getXMLElementValue(pRoot,"CAESAR/STREAM_SIZE")) >> m_caesarSettings.streamSize;
    pCaesarSettings = &m_caesarSettings;

    // bind project settings into global settings
    pGlobals->settings->project = (void*) pCaesarSettings;

    return STAT_OK;
}

int CaesarProject::initializeProject() {
    // allocate encryptor
    m_encryptor = new Encryptor;
    return STAT_OK;
}

int CaesarProject::initializeProjectState() {
    return STAT_NOT_IMPLEMENTED_YET;
}

int CaesarProject::saveProjectState(TiXmlNode* pRoot) const {
    int status = STAT_OK;
    TiXmlElement* pRoot2 = pRoot->ToElement();
    pRoot2->SetAttribute("loadable",0);
    return STAT_OK;
}

int CaesarProject::loadProjectState(TiXmlNode* pRoot) {
    return STAT_NOT_IMPLEMENTED_YET;
}

int CaesarProject::createTestVectorFilesHeaders() const {
    return STAT_NOT_IMPLEMENTED_YET;
}

int CaesarProject::generateTestVectors() {
    return STAT_NOT_IMPLEMENTED_YET;
}
