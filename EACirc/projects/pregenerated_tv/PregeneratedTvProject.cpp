#include "PregeneratedTvProject.h"

PregeneratedTvProject::PregeneratedTvProject()
    : IProject(PROJECT_PREGENERATED_TV) {
    m_tvFile.open(FILE_TEST_VECTORS);
    if (!m_tvFile.is_open()) {
        mainLogger.out() << "error: Cannot open file with pre-generated test vectors (" << FILE_TEST_VECTORS << ")!" << endl;
        return;
    }
}

PregeneratedTvProject::~PregeneratedTvProject() {
    if (m_tvFile.is_open()) {
        m_tvFile.close();
    }
}

string PregeneratedTvProject::shortDescription() const {
    return "loading pre-generated test vectors";
}

int PregeneratedTvProject::initializeProject() {
    if (!m_tvFile.is_open()) {
        return STAT_PROJECT_ERROR;
    }
    // checking settings
    bool error = false;
    int intSetting;

    // number of test sets
    m_tvFile >> intSetting;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    int requiredTestSets = pGlobals->settings->main.numGenerations / pGlobals->settings->testVectors.testVectorChangeFreq + 1;
    if (intSetting < requiredTestSets) {
        mainLogger.out() << "error: Not enough test sets in file." << endl;
        mainLogger.out() << "       required: " << requiredTestSets << endl;
        mainLogger.out() << "       provided: " << intSetting << endl;
        return STAT_PROJECT_ERROR;
    }
    // number of vectors in a set
    m_tvFile >> intSetting;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (intSetting != pGlobals->settings->testVectors.numTestVectors) {
        mainLogger.out() << "warning: Number of vectors in a set does not match.";
        mainLogger.out() << "         required: " << pGlobals->settings->testVectors.numTestVectors << endl;
        mainLogger.out() << "         provided: " << intSetting << endl;
        error = true;
    }
    // maximal number of inputs
    m_tvFile >> intSetting;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (intSetting != MAX_INPUTS) {
        mainLogger.out() << "warning: Maximal number of inputs does not match.";
        mainLogger.out() << "         required: " << MAX_INPUTS << endl;
        mainLogger.out() << "         provided: " << intSetting << endl;
        error = true;
    }
    // maximal number of outputs
    m_tvFile >> intSetting;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (intSetting != MAX_OUTPUTS) {
        mainLogger.out() << "warning: Maximal number of outputs does not match.";
        mainLogger.out() << "         required: " << MAX_OUTPUTS << endl;
        mainLogger.out() << "         provided: " << intSetting << endl;
        error = true;
    }
    // number of inputs
    m_tvFile >> intSetting;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (intSetting != pGlobals->settings->testVectors.testVectorLength) {
        mainLogger.out() << "warning: Number of inputs does not match.";
        mainLogger.out() << "         required: " << pGlobals->settings->testVectors.testVectorLength << endl;
        mainLogger.out() << "         provided: " << intSetting << endl;
        error = true;
    }
    // number of outputs
    m_tvFile >> intSetting;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (intSetting != pGlobals->settings->circuit.sizeOutputLayer) {
        mainLogger.out() << "warning: Number of outputs does not match.";
        mainLogger.out() << "         required: " << pGlobals->settings->circuit.sizeOutputLayer << endl;
        mainLogger.out() << "         provided: " << intSetting << endl;
        error = true;
    }

    // ignore project settings
    string line;
    do {
        getline(m_tvFile,line);
    } while (!line.empty());

    if (error) {
        mainLogger.out() << "warning: Settings in test vector file and project do not match." << endl;
    } else {
        mainLogger.out() << "info: Settings for project match." << endl;
    }

    // switch data read mode
    int dataPosition = m_tvFile.tellg();
    m_tvFile.close();
    m_tvFile.open(FILE_TEST_VECTORS, ios_base::binary);
    if (!m_tvFile.is_open()) {
        mainLogger.out() << "error: Cannot open file with pre-generated test vectors (" << FILE_TEST_VECTORS << ")!" << endl;
        return STAT_FILE_OPEN_FAIL;
    }
    m_tvFile.seekg(dataPosition);

    return STAT_OK;
}

int PregeneratedTvProject::generateTestVectors() {
    if (!m_tvFile.is_open()) {
        return STAT_PROJECT_ERROR;
    }
    for (int testVectorNumber = 0; testVectorNumber < pGlobals->settings->testVectors.numTestVectors; testVectorNumber++) {
        m_tvFile.read((char*)(pGlobals->testVectors[testVectorNumber]),MAX_INPUTS + MAX_OUTPUTS);
    }
    if (m_tvFile.fail()) {
        mainLogger.out() << "error: Problem when loading test vectors." << endl;
        return STAT_PROJECT_ERROR;
    }
    return STAT_OK;
}
