#include "IProject.h"
#include "CommonFnc.h"
#include "estream/EstreamProject.h"
#include "sha3/Sha3Project.h"

IProject::IProject(int type) : m_type(type) {}

IProject::~IProject() {}

int IProject::loadProjectConfiguration(TiXmlNode *pRoot) {
    return STAT_OK;
}

int IProject::initializeProject() {
    return STAT_OK;
}

TiXmlNode* IProject::saveProjectState() const {
    TiXmlElement* pNode = new TiXmlElement("project");
    pNode->SetAttribute("type",toString(m_type).c_str());
    pNode->SetAttribute("description",shortDescription().c_str());
    return pNode;
}

int IProject::initializeProjectState() {
    return STAT_OK;
}

int IProject::loadProjectState(TiXmlNode *pRoot) {
    int loadedType = atoi(getXMLElementValue(pRoot,"@type").c_str());
    if ( loadedType != m_type) {
        mainLogger.out() << "error: Incompatible project type." << endl;
        mainLogger.out() << "       required: " << m_type << "  given: " << loadedType << endl;
        return STAT_INCOMPATIBLE_PARAMETER;
    }
    return STAT_OK;
}

IProject* IProject::getProject(int projectType) {
    IProject* project = NULL;
    switch (projectType) {
    case PROJECT_ESTREAM:
        project = new EstreamProject();
        break;
    case PROJECT_SHA3:
        project = new Sha3Project();
        break;
    default:
        mainLogger.out() << "error: Cannot initialize project - unknown type (" << projectType << ")." << endl;
        return NULL;
        break;
    }
    mainLogger.out() << "info: Project successfully initialized. (" << project->shortDescription() << ")" << endl;
    return project;
}

int IProject::getProjectType() const {
    return m_type;
}

int IProject::generateAndSaveTestVectors() {
    int status = STAT_OK;
    if ((status = generateTestVectors()) != STAT_OK) {
        mainLogger.out() << "error: Test vector generation failed." << endl;
        return status;
    }
    if (pGlobals->settings->testVectors.saveTestVectors) {
        status = saveTestVectors();
    }
    return status;
}

int IProject::saveTestVectors() const {

    // TBD: save test vectors from pGlobals->testVectors to file in standardized format

    return STAT_OK;
}
