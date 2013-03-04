#include "Checker.h"
#include "EAC_circuit.h"

Checker::Checker()
    : m_status(STAT_OK) {
    if (pGlobals != NULL) {
        mainLogger.out(LOGGER_WARNING) << "Globals not NULL. Overwriting." << endl;
    }
    pGlobals = new GLOBALS;
    // CREATE STRUCTURE OF CIRCUIT FROM BASIC SETTINGS
    pGlobals->settings = &m_settings;
}

Checker::~Checker() {
    if (m_tvFile.is_open()) {
        m_tvFile.close();
    }
    if (pGlobals) {
        pGlobals->release();
        delete pGlobals;
    }
    pGlobals = NULL;
    if (rndGen) delete rndGen;
    rndGen = NULL;
    if (biasRndGen) delete biasRndGen;
    biasRndGen = NULL;
    if (galibGenerator) delete galibGenerator;
    galibGenerator = NULL;
    if (mainGenerator) delete mainGenerator;
    mainGenerator = NULL;
}

void Checker::setTestVectorFile(string filename) {
    if (m_status != STAT_OK) return;
    m_tvFilename = filename;
    m_tvFile.open(m_tvFilename);
    if (!m_tvFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot open file with pre-generated test vectors (" << m_tvFilename << ")." << endl;
        m_status = STAT_FILE_OPEN_FAIL;
        return;
    }
}

void Checker::loadTestVectorParameters() {
    if (m_status != STAT_OK) return;
    if (!m_tvFile.is_open()) {
        m_status = STAT_FILE_OPEN_FAIL;
        return;
    }

    // checking settings
    bool error = false;

    // number of test sets
    m_tvFile >> pGlobals->settings->testVectors.numTestSets;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (m_tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot read number of test sets." << endl;
        error = true;
    }
    // number of vectors in a set
    m_tvFile >> pGlobals->settings->testVectors.numTestVectors;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (m_tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot read number of test vectors in a set." << endl;
        error = true;
    }
    // maximal number of inputs
    m_tvFile >> m_max_inputs;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (m_tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot read maximal number of inputs." << endl;
        error = true;
    }
    // maximal number of outputs
    m_tvFile >> m_max_outputs;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (m_tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot read maximal number of outputs." << endl;
        error = true;
    }
    // number of inputs
    m_tvFile >> pGlobals->settings->testVectors.testVectorLength;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (m_tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot read test vector length." << endl;
        error = true;
    }
    // number of outputs
    m_tvFile >> pGlobals->settings->circuit.sizeOutputLayer;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (m_tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot read output layer size." << endl;
        error = true;
    }

    // ignore project settings
    string line;
    do {
        getline(m_tvFile,line);
    } while (!line.empty());

    if (error) {
        mainLogger.out(LOGGER_ERROR) << "Settings could not be read." << endl;
        m_status = STAT_CONFIG_DATA_READ_FAIL;
        return;
    } else {
        mainLogger.out(LOGGER_INFO) << "Settings successfully read." << endl;
    }

    // switch data read mode
    int dataPosition = m_tvFile.tellg();
    m_tvFile.close();
    m_tvFile.open(m_tvFilename, ios_base::binary);
    if (!m_tvFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot open file with pre-generated test vectors (" << m_tvFilename << ")!" << endl;
        m_status = STAT_FILE_OPEN_FAIL;
        return;
    }
    m_tvFile.seekg(dataPosition);
}

void Checker::check() {
    if (m_status != STAT_OK) return;

}

int Checker::getStatus() const {
    return m_status;
}
