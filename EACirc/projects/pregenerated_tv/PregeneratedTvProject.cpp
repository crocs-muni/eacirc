#include "PregeneratedTvProject.h"

PregeneratedTvProject::PregeneratedTvProject()
    : IProject(PROJECT_PREGENERATED_TV) {
    m_tvFile.open(FILE_TEST_VECTORS, ios_base::binary);
    if (!m_tvFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot open file with pre-generated test vectors (" << FILE_TEST_VECTORS << ")!" << endl;
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
    bool warning = false;
    int intSetting;

    // evaluator type
    m_tvFile >> intSetting;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    int requiredEvaluatorType = pGlobals->settings->main.evaluatorType;
    if (intSetting != requiredEvaluatorType) {
        mainLogger.out(LOGGER_WARNING) << "Incorrect evaluator used." << endl;
        mainLogger.out() << "         required: " << requiredEvaluatorType << endl;
        mainLogger.out() << "         provided: " << intSetting << endl;
        warning = true;
    }
    // number of test sets
    m_tvFile >> intSetting;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    int requiredTestSets = pGlobals->settings->testVectors.numTestSets + 1;
    if (intSetting < requiredTestSets) {
        mainLogger.out(LOGGER_ERROR) << "Not enough test sets in file." << endl;
        mainLogger.out() << "       required: " << requiredTestSets << endl;
        mainLogger.out() << "       provided: " << intSetting << endl;
        return STAT_PROJECT_ERROR;
    }
    // number of vectors in a set
    m_tvFile >> intSetting;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (intSetting != pGlobals->settings->testVectors.setSize) {
        mainLogger.out(LOGGER_WARNING) << "Number of vectors in a set does not match.";
        mainLogger.out() << "         required: " << pGlobals->settings->testVectors.setSize << endl;
        mainLogger.out() << "         provided: " << intSetting << endl;
        warning = true;
    }
    // number of inputs
    m_tvFile >> intSetting;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (intSetting != pGlobals->settings->testVectors.inputLength) {
        mainLogger.out(LOGGER_WARNING) << "Number of inputs does not match.";
        mainLogger.out() << "         required: " << pGlobals->settings->testVectors.inputLength << endl;
        mainLogger.out() << "         provided: " << intSetting << endl;
        warning = true;
    }
    // number of outputs
    m_tvFile >> intSetting;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (intSetting != pGlobals->settings->testVectors.outputLength) {
        mainLogger.out(LOGGER_WARNING) << "Number of outputs does not match.";
        mainLogger.out() << "         required: " << pGlobals->settings->circuit.sizeOutputLayer << endl;
        mainLogger.out() << "         provided: " << intSetting << endl;
        warning = true;
    }

    // ignore project settings
    // TBD/TODO: make better than engineering solution for line endings
    string line;
    do {
        getline(m_tvFile,line);
    } while (!line.empty() && line != "\r");

    if (warning) {
        mainLogger.out(LOGGER_WARNING) << "Settings in test vector file and project do not match." << endl;
    } else {
        mainLogger.out(LOGGER_INFO) << "Settings for project match." << endl;
    }
    return STAT_OK;
}

int PregeneratedTvProject::generateTestVectors() {
    if (!m_tvFile.is_open()) {
        return STAT_PROJECT_ERROR;
    }
    for (int testVectorNumber = 0; testVectorNumber < pGlobals->settings->testVectors.setSize; testVectorNumber++) {
        m_tvFile.read((char*)(pGlobals->testVectors.inputs[testVectorNumber]), pGlobals->settings->testVectors.inputLength);
        m_tvFile.read((char*)(pGlobals->testVectors.outputs[testVectorNumber]), pGlobals->settings->testVectors.outputLength);
    }
    if (m_tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Problem when loading test vectors." << endl;
        return STAT_PROJECT_ERROR;
    }
    return STAT_OK;
}
