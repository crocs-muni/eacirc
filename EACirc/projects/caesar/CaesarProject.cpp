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
    return "";
}

int CaesarProject::loadProjectConfiguration(TiXmlNode* pRoot) {
    return STAT_NOT_IMPLEMENTED_YET;
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
    return STAT_NOT_IMPLEMENTED_YET;
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
