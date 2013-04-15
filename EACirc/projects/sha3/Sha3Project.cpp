#include "Sha3Project.h"
#include "Sha3Constants.h"
#include "Sha3Interface.h"

SHA3_SETTINGS* pSha3Settings = NULL;

Sha3Project::Sha3Project()
    : IProject(PROJECT_SHA3), m_tvOutputs(NULL), m_tvInputs(NULL),
      m_numVectors(NULL), m_hasher(NULL) {
    m_tvOutputs = new unsigned char[pGlobals->settings->testVectors.outputLength];
    memset(m_tvOutputs,0,pGlobals->settings->testVectors.outputLength);
    m_tvInputs = new unsigned char[pGlobals->settings->testVectors.inputLength];
    memset(m_tvInputs,0,pGlobals->settings->testVectors.inputLength);
    m_numVectors = new int[2];
}

Sha3Project::~Sha3Project() {
    if (m_tvOutputs) delete[] m_tvOutputs;
    m_tvOutputs = NULL;
    if (m_tvInputs) delete[] m_tvInputs;
    m_tvInputs = NULL;
    if (m_numVectors) delete[] m_numVectors;
    m_numVectors = NULL;
    if (m_hasher) delete m_hasher;
    m_hasher = NULL;
}

string Sha3Project::shortDescription() const {
    return "SHA-3 candidate functions";
}

int Sha3Project::loadProjectConfiguration(TiXmlNode* pRoot) {
    m_sha3Settings.usageType = atoi(getXMLElementValue(pRoot,"SHA3/USAGE_TYPE").c_str());
    m_sha3Settings.plaintextType = atoi(getXMLElementValue(pRoot,"SHA3/PLAINTEXT_TYPE").c_str());
    m_sha3Settings.useFixedSeed = (atoi(getXMLElementValue(pRoot,"SHA3/USE_FIXED_SEED").c_str())) ? true : false;
    istringstream(getXMLElementValue(pRoot,"SHA3/SEED")) >> m_sha3Settings.seed;
    m_sha3Settings.algorithm1 = atoi(getXMLElementValue(pRoot,"SHA3/ALGORITHM_1").c_str());
    m_sha3Settings.algorithm2 = atoi(getXMLElementValue(pRoot,"SHA3/ALGORITHM_2").c_str());
    m_sha3Settings.hashLength1 = atoi(getXMLElementValue(pRoot,"SHA3/HASHLENGTH_ALG_1").c_str());
    m_sha3Settings.hashLength2 = atoi(getXMLElementValue(pRoot,"SHA3/HASHLENGTH_ALG_2").c_str());
    m_sha3Settings.ballancedTestVectors = (atoi(getXMLElementValue(pRoot,"SHA3/BALLANCED_TEST_VECTORS").c_str())) ? true : false;
    m_sha3Settings.limitAlgRounds = (atoi(getXMLElementValue(pRoot,"SHA3/LIMIT_NUM_OF_ROUNDS").c_str())) ? true : false;
    m_sha3Settings.alg1RoundsCount = atoi(getXMLElementValue(pRoot,"SHA3/ROUNDS_ALG_1").c_str());
    m_sha3Settings.alg2RoundsCount = atoi(getXMLElementValue(pRoot,"SHA3/ROUNDS_ALG_2").c_str());
    m_sha3Settings.generateStream = (atoi(getXMLElementValue(pRoot,"SHA3/GENERATE_STREAM").c_str())) ? true : false;
    istringstream ss(getXMLElementValue(pRoot,"SHA3/STREAM_SIZE"));
    ss >> m_sha3Settings.streamSize;
    pSha3Settings = &m_sha3Settings;

    // bind project settings into global settings
    pGlobals->settings->project = (void*) pSha3Settings;

    return STAT_OK;
}

int Sha3Project::initializeProject() {
    // allocate hasher
    m_hasher = new Hasher;
    return STAT_OK;
}

int Sha3Project::initializeProjectState() {
    return m_hasher->initializeState();
}

int Sha3Project::createTestVectorFilesHeaders() const {
    // generate header (project config) to test vector file
    ofstream tvFile;
    tvFile.open(FILE_TEST_VECTORS, ios_base::app | ios_base::binary);
    if (!tvFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    tvFile << pGlobals->settings->main.projectType << " \t\t(project: " << shortDescription() << ")" << endl;
    tvFile << pSha3Settings->usageType << " \t\t(usage type)" << endl;
    tvFile << pSha3Settings->plaintextType << " \t\t(plaintext type)" << endl;
    tvFile << pSha3Settings->algorithm1 << " \t\t(algorithm1: " << Sha3Interface::sha3ToString(pSha3Settings->algorithm1) << ")" << endl;
    tvFile << pSha3Settings->algorithm2 << " \t\t(algorithm2: " << Sha3Interface::sha3ToString(pSha3Settings->algorithm2) << ")" << endl;
    tvFile << pSha3Settings->hashLength1 << " \t\t(hash length for algorithm1)" << endl;
    tvFile << pSha3Settings->hashLength2 << " \t\t(hash length for algorithm2)" << endl;
    tvFile << pSha3Settings->seed << " \t\t(initial seed for counters)" << endl;
    tvFile << pSha3Settings->ballancedTestVectors << " \t\t(ballanced test vectors?)" << endl;
    tvFile << pSha3Settings->limitAlgRounds << " \t\t(limit algorithm rounds?)" << endl;
    if (pSha3Settings->limitAlgRounds) {
        tvFile << pSha3Settings->alg1RoundsCount << " \t\t(algorithm1: " << pSha3Settings->alg1RoundsCount << " rounds)" << endl;
        tvFile << pSha3Settings->alg2RoundsCount << " \t\t(algorithm2: " << pSha3Settings->alg2RoundsCount << " rounds)" << endl;
    }
    tvFile.close();

    // generate header to human-readable test-vector file
    tvFile.open(FILE_TEST_VECTORS_HR, ios::app | ios_base::binary);
    if (!tvFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    tvFile << "Using SHA-3 hash functions and random generator to generate test vectors." << endl;
    tvFile << "  stream1: using " << Sha3Interface::sha3ToString(pSha3Settings->algorithm1);
    if (pSha3Settings->limitAlgRounds) {
        tvFile << " (" << pSha3Settings->alg1RoundsCount << " rounds)" << endl;
    } else {
        tvFile << " (unlimited version)" << endl;
    }
    tvFile << "  stream2: using " << Sha3Interface::sha3ToString(pSha3Settings->algorithm2);
    if (pSha3Settings->limitAlgRounds) {
        tvFile << " (" << pSha3Settings->alg2RoundsCount << " rounds)" << endl;
    } else {
        tvFile << " (unlimited version)" << endl;
    }
    tvFile << "Test vectors formatted as INPUT::OUTPUT" << endl;
    tvFile << endl;
    tvFile.close();

    return STAT_OK;
}

int Sha3Project::generateTestVectors() {
    int status = STAT_OK;

    // if set so, do not generate test vectors but generate data stream to cout
    if (pSha3Settings->generateStream) {
        status = generateHashDataStream();
        if (status != STAT_OK) {
            return status;
        } else {
            return STAT_INTENTIONAL_EXIT;
        }
    }

    // USED FOR BALANCING TEST VECTORS
    this->m_numVectors[0] = 0;
    this->m_numVectors[1] = 0;

    for (int testVectorNumber = 0; testVectorNumber < pGlobals->settings->testVectors.setSize; testVectorNumber++) {
        if (pGlobals->settings->testVectors.saveTestVectors == 1) {
            ofstream tvfile(FILE_TEST_VECTORS_HR, ios::app);
            tvfile << "Test vector n." << dec << testVectorNumber << endl;
            tvfile.close();
        }

        status = prepareSingleTestVector();
        if (status != STAT_OK) {
            return status;
        }

        for (int inputByte = 0; inputByte < pGlobals->settings->testVectors.inputLength; inputByte++) {
            pGlobals->testVectors.inputs[testVectorNumber][inputByte] = m_tvInputs[inputByte];
        }
        for (int outputByte = 0; outputByte < pGlobals->settings->testVectors.outputLength; outputByte++)
            pGlobals->testVectors.outputs[testVectorNumber][outputByte] = m_tvOutputs[outputByte];
    }
    return status;
}

int Sha3Project::prepareSingleTestVector() {
    int status = STAT_OK;
    //! are we using algorithm1 (0) or algorithm2 (1) ?
    int algorithmNumber = 0;
    switch (pSha3Settings->usageType) {
    case SHA3_DISTINGUISHER:
        //SHALL WE BALANCE TEST VECTORS?
        if (pSha3Settings->ballancedTestVectors && (m_numVectors[0] >= pGlobals->settings->testVectors.setSize/2))
            algorithmNumber = 1;
        else if (pSha3Settings->ballancedTestVectors && (m_numVectors[1] >= pGlobals->settings->testVectors.setSize/2))
            algorithmNumber = 0;
        else
            rndGen->getRandomFromInterval(1, &algorithmNumber);
        m_numVectors[algorithmNumber]++;
        status = m_hasher->getTestVector(algorithmNumber,m_tvInputs,m_tvOutputs);
        break;
    default:
        mainLogger.out(LOGGER_ERROR) << "Unknown usage type (" << pSha3Settings->usageType << ")." << endl;
        return STAT_INVALID_ARGUMETS;
        break;
    }

    // save human-readable test vector
    if (pGlobals->settings->testVectors.saveTestVectors) {
        ofstream tvFile(FILE_TEST_VECTORS_HR, ios::app);
        tvFile << setfill('0');
        for (int input = 0; input < pGlobals->settings->testVectors.inputLength; input++)
            tvFile << setw(2) << hex << (int)(m_tvInputs[input]);
        tvFile << "::";
        for (int output = 0; output < pGlobals->settings->testVectors.outputLength; output++)
            tvFile << setw(2) << hex << (int)(m_tvOutputs[output]);
        tvFile << endl;
        tvFile.close();
    }
    return status;
}

int Sha3Project::saveProjectState(TiXmlNode* pRoot) const {
    int status = STAT_OK;
    TiXmlElement* pRoot2 = pRoot->ToElement();
    pRoot2->SetAttribute("loadable",1);
    TiXmlElement* pElem = new TiXmlElement("hasher");
    status =  m_hasher->saveHasherState(pElem);
    pRoot2->LinkEndChild(pElem);
    return status;
}

int Sha3Project::loadProjectState(TiXmlNode* pRoot) {
    return m_hasher->loadHasherState(getXMLElement(pRoot,"hasher"));
}

int Sha3Project::generateHashDataStream() {
    int status = STAT_OK;
    int algorithm = -1;
    int numRounds = -1;
    string streamFilename;
    for (int algorithmNumber = 0; algorithmNumber < 2; algorithmNumber++) {
        switch (algorithmNumber) {
        case 0:
            algorithm = pSha3Settings->algorithm1;
            numRounds = pSha3Settings->limitAlgRounds ? pSha3Settings->alg1RoundsCount : -1;
            streamFilename = SHA3_FILE_STREAM_1;
            break;
        case 1:
            algorithm = pSha3Settings->algorithm2;
            numRounds = pSha3Settings->limitAlgRounds ? pSha3Settings->alg2RoundsCount : -1;
            streamFilename = SHA3_FILE_STREAM_2;
            break;
        default:
            mainLogger.out(LOGGER_ERROR) << "Unsupported iteration while generating testVector streams (";
            mainLogger.out() << algorithmNumber << ")." << endl;
            return STAT_PROJECT_ERROR;
        }
        if (algorithm == SHA3_RANDOM) {
            mainLogger.out(LOGGER_INFO) << "Algorithm " << (algorithmNumber+1);
            mainLogger.out() << " is set to random, stream data generation skipped." << endl;
            continue;
        } else {
            mainLogger.out(LOGGER_INFO) << "Generating stream for " << Sha3Interface::sha3ToString(algorithm);
            if (numRounds == -1) {
                mainLogger.out() << " (unlimitted version)." << endl;
            } else {
                mainLogger.out() << " (" << numRounds << " rounds)." << endl;
            }
            mainLogger.out(LOGGER_INFO) << "Output is saved to file \"" << streamFilename << "\"." << endl;
        }
        ostream* vectorStream = NULL;
        if (pSha3Settings->streamSize == 0) {
            vectorStream = &cout;
        } else {
            vectorStream = new ofstream(streamFilename, ios_base::binary | ios_base::trunc);
        }
        unsigned long alreadyGenerated = 0;
        while (pSha3Settings->streamSize == 0 ? true : alreadyGenerated <= pSha3Settings->streamSize) {
            for (int testVectorNumber = 0; testVectorNumber < pGlobals->settings->testVectors.setSize; testVectorNumber++) {
                if (status != STAT_OK) break;

                // get test vector for algorithmNumber
                status = m_hasher->getTestVector(algorithmNumber,m_tvInputs,m_tvOutputs);

                for (int index = 0; index < pGlobals->settings->testVectors.inputLength; index++) {
                    (*vectorStream) << m_tvInputs[index];
                }
                alreadyGenerated += pGlobals->settings->testVectors.inputLength;
            }
        }

        if (pSha3Settings->streamSize != 0) {
            delete vectorStream;
            vectorStream = NULL;
        }
        mainLogger.out(LOGGER_INFO) << "Hash data generation ended (" << alreadyGenerated << " bytes)." << endl;
    }

    return status;
}
