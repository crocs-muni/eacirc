#include "IProject.h"
#include "CommonFnc.h"
#include "pregenerated_tv/PregeneratedTvProject.h"
#include "estream/EstreamProject.h"
#include "sha3/Sha3Project.h"

IProject::IProject(int type) : m_type(type) {
    if (pGlobals->settings->testVectors.saveTestVectors && pGlobals->settings->main.projectType != PROJECT_PREGENERATED_TV) {
        ofstream tvFile;
        tvFile.open(FILE_TEST_VECTORS, ios_base::trunc);
        if (!tvFile.is_open()) {
            mainLogger.out() << "error: Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
            return;
        }
        tvFile << dec << left;
        tvFile << (pGlobals->settings->main.numGenerations / pGlobals->settings->testVectors.testVectorChangeFreq + 1);
        tvFile << " \t\t(number of test vector sets)" << endl;
        tvFile << pGlobals->settings->testVectors.numTestVectors << " \t\t(number of test vectors in a set)" << endl;
        tvFile << MAX_INPUTS << " \t\t(maximal number of inputs)" << endl;
        tvFile << MAX_OUTPUTS << " \t\t(maximal number of outputs)" << endl;
        tvFile << pGlobals->settings->testVectors.testVectorLength << " \t\t(number of tv input bytes)" << endl;
        tvFile << pGlobals->settings->circuit.sizeOutputLayer << " \t\t(number of tv output bytes)" << endl;
        tvFile.close();
    }
}

IProject::~IProject() {}

int IProject::loadProjectConfiguration(TiXmlNode *pRoot) {
    return STAT_OK;
}

int IProject::initializeProject() {
    return STAT_OK;
}

int IProject::initializeProjectMain() {
    int status = STAT_OK;
    status = initializeProject();
    if (status != STAT_OK) return status;
    if (pGlobals->settings->testVectors.saveTestVectors && pGlobals->settings->main.projectType != PROJECT_PREGENERATED_TV) {
        ofstream tvFile;
        tvFile.open(FILE_TEST_VECTORS, ios_base::app);
        if (!tvFile.is_open()) {
            mainLogger.out() << "error: Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
            return STAT_FILE_WRITE_FAIL;
        }
        tvFile << endl;
        tvFile.close();
    }
    return status;
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
    case PROJECT_PREGENERATED_TV:
        project = new PregeneratedTvProject();
        break;
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
    ofstream tvFile;
    tvFile.open(FILE_TEST_VECTORS, ios_base::app | ios_base::binary);
    if (!tvFile.is_open()) {
        mainLogger.out() << "error: Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    for (int testVectorNumber = 0; testVectorNumber < pGlobals->settings->testVectors.numTestVectors; testVectorNumber++) {
        tvFile.write((char*)(pGlobals->testVectors[testVectorNumber]),MAX_INPUTS + MAX_OUTPUTS);
    }
    if (tvFile.fail()) {
        mainLogger.out() << "error: Problem when saving test vectors." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    tvFile.close();
    return STAT_OK;
}
