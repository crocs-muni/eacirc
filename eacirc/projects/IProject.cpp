#include "IProject.h"
#include "CommonFnc.h"
#include "evaluators/IEvaluator.h"
#include "pregenerated_tv/PregeneratedTvProject.h"
#include "files/filesProject.h"

#ifdef ESTREAM
    #include "estream/EstreamProject.h"
#endif

#ifdef SHA3
    #include "sha3/Sha3Project.h"
#endif

#ifdef CAESAR
    #include "caesar/CaesarProject.h"
#endif


IProject::IProject(int type) : m_type(type), m_projectEvaluator(NULL) { }

IProject::~IProject() {
    if (m_projectEvaluator != NULL) {
        delete m_projectEvaluator;
        m_projectEvaluator = NULL;
    }
}

int IProject::loadProjectConfiguration(TiXmlNode *pRoot) {
    return STAT_OK;
}

int IProject::initializeProject() {
    return STAT_OK;
}

int IProject::initializeProjectState() {
    return STAT_OK;
}

TiXmlNode* IProject::saveProjectStateMain() const {
    TiXmlElement* pNode = new TiXmlElement("project");
    pNode->SetAttribute("type",CommonFnc::toString(m_type).c_str());
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

int IProject::createTestVectorFilesHeaders() const {
    return STAT_OK;
}

int IProject::createTestVectorFilesHeadersMain() const {
    int status = STAT_OK;
    ofstream tvFile;
    if (pGlobals->settings->outputs.saveTestVectors) {
        tvFile.open(FILE_TEST_VECTORS, ios_base::trunc | ios_base::binary);
        if (!tvFile.is_open()) {
            mainLogger.out(LOGGER_ERROR) << "Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
            return STAT_FILE_WRITE_FAIL;
        }
        tvFile << dec << left;
        tvFile << pGlobals->settings->main.evaluatorType << " \t\t(evaluator: " << pGlobals->evaluator->shortDescription() << ")" << endl;
        tvFile << pGlobals->settings->testVectors.numTestSets + 1 << " \t\t(number of test vector sets)" << endl;
        tvFile << pGlobals->settings->testVectors.setSize << " \t\t(number of test vectors in a set)" << endl;
        tvFile << pGlobals->settings->testVectors.inputLength << " \t\t(number of tv input bytes)" << endl;
        tvFile << pGlobals->settings->testVectors.outputLength << " \t\t(number of tv output bytes)" << endl;
        tvFile.close();
    }

    status = createTestVectorFilesHeaders();
    if (status != STAT_OK) return status;

    if (pGlobals->settings->outputs.saveTestVectors) {
        tvFile.open(FILE_TEST_VECTORS, ios_base::app | ios_base::binary);
        if (!tvFile.is_open()) {
            mainLogger.out(LOGGER_ERROR) << "Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
            return STAT_FILE_WRITE_FAIL;
        }
        tvFile << endl; // separator between header and test vector values
        tvFile.close();
    }

    return status;
}

int IProject::generateAndSaveTestVectors() {
    int status = STAT_OK;
    if (pGlobals->settings->outputs.verbosity >= LOGGER_VERBOSITY_DEVELOPMENT) {
        mainLogger.out(LOGGER_INFO) << "Generating test vectors." << endl;
    }
    if ((status = generateTestVectors()) != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Test vector generation failed." << endl;
        return status;
    }
    pGlobals->testVectors.newSet = true;
    status = saveTestVectors();

    return status;
}

int IProject::saveTestVectors() const {
    ofstream tvFile;

    // save binary form
    if (pGlobals->settings->outputs.saveTestVectors) {
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
            mainLogger.out(LOGGER_ERROR) << "Problem when saving test vectors (" << FILE_TEST_VECTORS << ")." << endl;
            return STAT_FILE_WRITE_FAIL;
        }
        tvFile.close();
    }

    // save human-readable form
    if (pGlobals->settings->outputs.verbosity >= LOGGER_VERBOSITY_DEBUG) {
        tvFile.open(FILE_TEST_VECTORS_HR, ios_base::app);
        if (!tvFile.is_open()) {
            mainLogger.out(LOGGER_ERROR) << "Cannot write file for test vectors (" << FILE_TEST_VECTORS_HR << ")." << endl;
            return STAT_FILE_WRITE_FAIL;
        }
        tvFile << endl << "##### Final Test set for generation " << pGlobals->stats.actGener << " #####" << endl;
        for (int testVector = 0; testVector < pGlobals->settings->testVectors.setSize; testVector++) {
            tvFile << "in:  ";
            for (int byte = 0; byte < pGlobals->settings->testVectors.inputLength; byte++) {
                tvFile << hex << setfill('0') << setw(2) << (int) pGlobals->testVectors.inputs[testVector][byte];
            }
            tvFile << endl;
            tvFile << "out: ";
            for (int byte = 0; byte < pGlobals->settings->testVectors.outputLength; byte++) {
                tvFile << hex << setfill('0') << setw(2) << (int) pGlobals->testVectors.outputs[testVector][byte];
            }
            tvFile << endl;
        }
        if (tvFile.fail()) {
            mainLogger.out(LOGGER_ERROR) << "Problem when saving test vectors (" << FILE_TEST_VECTORS_HR << ")." << endl;
            return STAT_FILE_WRITE_FAIL;
        }
        tvFile.close();
    }

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
#ifdef ESTREAM
    case PROJECT_ESTREAM:
        project = new EstreamProject();
        break;
#endif
#ifdef SHA3
    case PROJECT_SHA3:
        project = new Sha3Project();
        break;
#endif
    case PROJECT_FILE_DISTINGUISHER:
        project = new FilesProject();
        break;
#ifdef CAESAR
    case PROJECT_CAESAR:
        project = new CaesarProject();
        break;
#endif
    default:
        mainLogger.out(LOGGER_ERROR) << "Cannot initialize project - unknown type (" << projectType << ")." << endl;
        return NULL;
        break;
    }
    mainLogger.out(LOGGER_INFO) << "Project successfully initialized. (" << project->shortDescription() << ")" << endl;
    return project;
}

string IProject::getTestingConfiguration(int projectType) {
    string projectConfiguration;
    switch (projectType) {
    case PROJECT_PREGENERATED_TV:
        projectConfiguration = "<PROJECT />";
        break;
#ifdef ESTREAM
    case PROJECT_ESTREAM:
        projectConfiguration = EstreamProject::testingConfiguration();
        break;
#endif
#ifdef SHA3
    case PROJECT_SHA3:
        projectConfiguration = Sha3Project::testingConfiguration();
        break;
#endif
    case PROJECT_FILE_DISTINGUISHER:
        projectConfiguration = FilesProject::testingConfiguration();
        break;
#ifdef CAESAR
    case PROJECT_CAESAR:
        projectConfiguration = CaesarProject::testingConfiguration();
        break;
#endif
    default:
        mainLogger.out(LOGGER_ERROR) << "No configuration - unknown project type (" << projectType << ")." << endl;
        return "";
        break;
    }
    return projectConfiguration;
}
