#include "IProject.h"
#include "CommonFnc.h"
#include "evaluators/IEvaluator.h"
#include "pregenerated_tv/PregeneratedTvProject.h"
#include "estream/EstreamProject.h"
#include "sha3/Sha3Project.h"
#include "tea/TeaProject.h"

IProject::IProject(int type) : m_type(type), m_projectEvaluator(NULL) {
    if (pGlobals->settings->testVectors.saveTestVectors && pGlobals->settings->main.projectType != PROJECT_PREGENERATED_TV) {
        ofstream tvFile;
        tvFile.open(FILE_TEST_VECTORS, ios_base::trunc | ios_base::binary);
        if (!tvFile.is_open()) {
            mainLogger.out(LOGGER_ERROR) << "Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
            return;
        }
        tvFile << dec << left;
        tvFile << pGlobals->settings->main.evaluatorType << " \t\t(evaluator";
        IEvaluator* evaluator = IEvaluator::getStandardEvaluator(pGlobals->settings->main.evaluatorType);
        if (evaluator) tvFile << ": " << evaluator->shortDescription();
        delete evaluator;
        tvFile << ")" << endl;
        tvFile << pGlobals->settings->testVectors.numTestSets + 1 << " \t\t(number of test vector sets)" << endl;
        tvFile << pGlobals->settings->testVectors.setSize << " \t\t(number of test vectors in a set)" << endl;
        tvFile << pGlobals->settings->testVectors.inputLength << " \t\t(number of tv input bytes)" << endl;
        tvFile << pGlobals->settings->testVectors.outputLength << " \t\t(number of tv output bytes)" << endl;
        tvFile.close();
    }
}

IProject::~IProject() {
    if (m_projectEvaluator != NULL) {
        delete m_projectEvaluator;
        m_projectEvaluator = NULL;
    }
}

int IProject::loadProjectConfiguration(TiXmlNode *pRoot) {
    return STAT_OK;
}

int IProject::initializeProjectMain() {
    int status = STAT_OK;
    status = initializeProject();
    if (status != STAT_OK) return status;
    if (pGlobals->settings->testVectors.saveTestVectors && pGlobals->settings->main.projectType != PROJECT_PREGENERATED_TV) {
        ofstream tvFile;
        tvFile.open(FILE_TEST_VECTORS, ios_base::app | ios_base::binary);
        if (!tvFile.is_open()) {
            mainLogger.out(LOGGER_ERROR) << "Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
            return STAT_FILE_WRITE_FAIL;
        }
        tvFile << endl;
        tvFile.close();
    }
    return status;
}

int IProject::initializeProject() {
    return STAT_OK;
}

int IProject::initializeProjectState() {
    return STAT_OK;
}

TiXmlNode* IProject::saveProjectStateMain() const {
    TiXmlElement* pNode = new TiXmlElement("project");
    pNode->SetAttribute("type",toString(m_type).c_str());
    pNode->SetAttribute("description",shortDescription().c_str());
    pNode->SetAttribute("loadable",0);
    if (saveProjectState(pNode) != STAT_OK) {
        mainLogger.out(LOGGER_WARNING) << "Project status could not be saved." << endl;
    }
    return pNode;
}

int IProject::saveProjectState(TiXmlNode* pRoot) const {
    return STAT_OK;
}

int IProject::loadProjectStateMain(TiXmlNode *pRoot) {
    int loadedType = atoi(getXMLElementValue(pRoot,"@type").c_str());
    if (loadedType != m_type) {
        mainLogger.out(LOGGER_ERROR) << "Incompatible project type." << endl;
        mainLogger.out() << "       required: " << m_type << "  given: " << loadedType << endl;
        return STAT_INVALID_ARGUMETS;
    }
    bool loadable = atoi(getXMLElementValue(pRoot,"@loadable").c_str()) ? true : false;
    if (!loadable) {
        mainLogger.out(LOGGER_ERROR) << "Project is set as unloadable." << endl;
        return STAT_INVALID_ARGUMETS;
    }
    if (loadProjectState(pRoot) != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Project state could not be loaded." << endl;
        return STAT_PROJECT_ERROR;
    }
    return STAT_OK;
}

int IProject::loadProjectState(TiXmlNode* pRoot) {
    return STAT_OK;
}

int IProject::getProjectType() const {
    return m_type;
}

int IProject::generateAndSaveTestVectors() {
    int status = STAT_OK;
    mainLogger.out(LOGGER_INFO) << "Generating test vectors." << endl;
    if ((status = generateTestVectors()) != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Test vector generation failed." << endl;
        return status;
    }
    pGlobals->testVectors.newSet = true;
    if (pGlobals->settings->testVectors.saveTestVectors && pGlobals->settings->main.projectType != PROJECT_PREGENERATED_TV) {
        status = saveTestVectors();
    }
    return status;
}

int IProject::saveTestVectors() const {
    ofstream tvFile;
    tvFile.open(FILE_TEST_VECTORS, ios_base::app | ios_base::binary);
    if (!tvFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    for (int testVector = 0; testVector < pGlobals->settings->testVectors.setSize; testVector++) {
        tvFile.write((char*)(pGlobals->testVectors.inputs[testVector]), pGlobals->settings->testVectors.inputLength);
        tvFile.write((char*)(pGlobals->testVectors.outputs[testVector]), pGlobals->settings->testVectors.outputLength);
    }
    if (tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Problem when saving test vectors." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    tvFile.close();
    return STAT_OK;
}

IEvaluator* IProject::getProjectEvaluator() {
    return m_projectEvaluator;
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
    case PROJECT_TEA:
        project = new TeaProject();
        break;
    default:
        mainLogger.out(LOGGER_ERROR) << "Cannot initialize project - unknown type (" << projectType << ")." << endl;
        return NULL;
        break;
    }
    mainLogger.out(LOGGER_INFO) << "Project successfully initialized. (" << project->shortDescription() << ")" << endl;
    return project;
}
