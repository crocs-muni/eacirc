#include "Checker.h"
#include "EAC_circuit.h"

Checker::Checker()
    : m_status(STAT_OK), m_evaluator(NULL) {
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
    if (m_evaluator) delete m_evaluator;
    m_evaluator = NULL;
    if (pGlobals) {
        pGlobals->release();
        delete pGlobals;
    }
    pGlobals = NULL;
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

    // evaluator
    m_tvFile >> pGlobals->settings->main.evaluatorType;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (m_tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot read evaluator type." << endl;
        error = true;
    }

    // number of test sets
    m_tvFile >> pGlobals->settings->testVectors.numTestSets;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (m_tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot read number of test sets." << endl;
        error = true;
    }
    // number of vectors in a set
    m_tvFile >> pGlobals->settings->testVectors.setSize;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (m_tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot read number of test vectors in a set." << endl;
        error = true;
    }
    // number of inputs
    m_tvFile >> pGlobals->settings->testVectors.inputLength;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (m_tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot read test vector input length." << endl;
        error = true;
    }
    // number of outputs
    m_tvFile >> pGlobals->settings->testVectors.outputLength;
    m_tvFile.ignore(UCHAR_MAX,'\n');
    if (m_tvFile.fail()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot read test vector output length." << endl;
        error = true;
    }
    // TBD/TODO: only temporary solution
    pGlobals->settings->circuit.sizeOutputLayer = pGlobals->settings->testVectors.outputLength;

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

    // load and allocate resources
    pGlobals->allocate();
    m_evaluator = IEvaluator::getEvaluator(pGlobals->settings->main.evaluatorType);
    if (m_evaluator == NULL) m_status = STAT_INVALID_ARGUMETS;
}

void Checker::check() {
    if (m_status != STAT_OK) return;

    unsigned char* circuitOutputs;
    circuitOutputs = new unsigned char[pGlobals->settings->circuit.sizeOutputLayer];
    int totalMatched = 0;
    int totalPredictions = 0;
    int setMatched;
    int setPredictions;
    double fitness;

    ofstream fitProgressFile;
    fitProgressFile.open(FILE_FITNESS_PROGRESS);
    if (!fitProgressFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write file for fitness progress (" << FILE_FITNESS_PROGRESS << ")." << endl;
        m_status = STAT_FILE_WRITE_FAIL;
        return;
    }

    for (int testSet = 0; testSet < pGlobals->settings->testVectors.numTestSets; testSet++) {
        // clear statistics
        setMatched = 0;
        setPredictions = 0;
        // read test set from file
        for (int testVector = 0; testVector < pGlobals->settings->testVectors.setSize; testVector++) {
            m_tvFile.read((char*)(pGlobals->testVectors[testVector]),
                          pGlobals->settings->testVectors.inputLength + pGlobals->settings->testVectors.outputLength);
        }
        // run circuits on test set
        for (int testVector = 0; testVector < pGlobals->settings->testVectors.setSize; testVector++) {
            circuit(pGlobals->testVectors[testVector],circuitOutputs);
            m_evaluator->evaluateCircuit(circuitOutputs,pGlobals->testVectors[testVector] +
                                         pGlobals->settings->testVectors.inputLength,NULL,&setMatched,&setPredictions);
        }

        fitness = setPredictions != 0 ? (double) setMatched / setPredictions : 0;
        totalMatched += setMatched;
        totalPredictions += setPredictions;
        fitProgressFile << testSet << "\t" << fitness << "\t" << setMatched << "\t" << setPredictions << endl;
    }

    fitness = totalPredictions != 0 ? (double) totalMatched / totalPredictions : 0;
    fitProgressFile << endl;
    fitProgressFile << "total:\t" << fitness << "\t" << totalMatched << "\t" << totalPredictions << endl;
    mainLogger.out(LOGGER_INFO) << "Static check finished successfully (average fitness: " << fitness << " )." <<  endl;

    delete circuitOutputs;
    fitProgressFile.close();
}

int Checker::getStatus() const {
    return m_status;
}
