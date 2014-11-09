#include "CaesarProject.h"

CAESAR_SETTINGS* pCaesarSettings = NULL;

CaesarProject::CaesarProject()
    : IProject(PROJECT_CAESAR), m_encryptor(NULL), m_ciphertext(NULL), m_realCiphertextLength(0) { }

CaesarProject::~CaesarProject() {
    if (m_encryptor != NULL) { delete m_encryptor; m_encryptor = NULL; }
    if (m_ciphertext != NULL) { delete m_ciphertext; m_ciphertext = NULL; }
}

string CaesarProject::shortDescription() const {
    return "CAESAR -- competition for authenticated encryption";
}

string CaesarProject::testingConfiguration() {
    string config =
            "<CAESAR>"
            "    <USAGE_TYPE>301</USAGE_TYPE>"
            "    <ALGORITHM>1</ALGORITHM>"
            "    <LIMIT_NUM_OF_ROUNDS>0</LIMIT_NUM_OF_ROUNDS>"
            "    <ALGORITHM_ROUNDS>3</ALGORITHM_ROUNDS>"
            "    <PLAINTEXT_LENGTH>16</PLAINTEXT_LENGTH>"
            "    <AD_LENGTH>0</AD_LENGTH>"
            "    <PLAINTEXT_TYPE>0</PLAINTEXT_TYPE>"
            "    <KEY_TYPE>2</KEY_TYPE>"
            "    <AD_TYPE>0</AD_TYPE>"
            "    <SMN_TYPE>0</SMN_TYPE>"
            "    <PMN_TYPE>0</PMN_TYPE>"
            "    <GENERATE_STREAM>0</GENERATE_STREAM>"
            "    <STREAM_SIZE>5242880</STREAM_SIZE>"
            "</CAESAR>";
    return config;
}

int CaesarProject::loadProjectConfiguration(TiXmlNode* pRoot) {
    m_caesarSettings.usageType = atoi(getXMLElementValue(pRoot,"CAESAR/USAGE_TYPE").c_str());
    m_caesarSettings.algorithm = atoi(getXMLElementValue(pRoot,"CAESAR/ALGORITHM").c_str());
    m_caesarSettings.limitAlgRounds = (atoi(getXMLElementValue(pRoot,"CAESAR/LIMIT_NUM_OF_ROUNDS").c_str())) ? true : false;
    m_caesarSettings.algorithmRoundsCount = atoi(getXMLElementValue(pRoot,"CAESAR/ALGORITHM_ROUNDS").c_str());
    istringstream(getXMLElementValue(pRoot,"CAESAR/PLAINTEXT_LENGTH")) >> m_caesarSettings.plaintextLength;
    istringstream(getXMLElementValue(pRoot,"CAESAR/AD_LENGTH")) >> m_caesarSettings.adLength;
    m_caesarSettings.plaintextType = atoi(getXMLElementValue(pRoot,"CAESAR/PLAINTEXT_TYPE").c_str());
    m_caesarSettings.keyType = atoi(getXMLElementValue(pRoot,"CAESAR/KEY_TYPE").c_str());
    m_caesarSettings.adType = atoi(getXMLElementValue(pRoot,"CAESAR/AD_TYPE").c_str());
    m_caesarSettings.smnType = atoi(getXMLElementValue(pRoot,"CAESAR/SMN_TYPE").c_str());
    m_caesarSettings.pmnType = atoi(getXMLElementValue(pRoot,"CAESAR/PMN_TYPE").c_str());
    m_caesarSettings.generateStream = (atoi(getXMLElementValue(pRoot,"CAESAR/GENERATE_STREAM").c_str())) ? true : false;
    istringstream(getXMLElementValue(pRoot,"CAESAR/STREAM_SIZE")) >> m_caesarSettings.streamSize;
    pCaesarSettings = &m_caesarSettings;

    // adjust number of rounds if necessary
    if (!m_caesarSettings.limitAlgRounds) { m_caesarSettings.algorithmRoundsCount = -1; }

    // bind project settings into global settings
    pGlobals->settings->project = (void*) pCaesarSettings;

    // configuration checks
    // TODO: do somthing with signed/unsigned comparison
    if (pGlobals->settings->testVectors.inputLength != m_caesarSettings.plaintextLength) {
        mainLogger.out(LOGGER_ERROR) << "Test vector input length does not match plaintext length!" << endl;
        return STAT_PROJECT_ERROR;
    }

    return STAT_OK;
}

int CaesarProject::initializeProject() {
    // allocate encryptor
    m_encryptor = new Encryptor;
    // allocate ciphertext buffer
    m_ciphertext = new bits_t[pCaesarSettings->ciphertextLength];

    // TODO create headers for human readable test vector file

    return STAT_OK;
}

int CaesarProject::initializeProjectState() {
    return m_encryptor->setup();
}

int CaesarProject::saveProjectState(TiXmlNode* pRoot) const {
    int status = STAT_OK;
    TiXmlElement* pRoot2 = pRoot->ToElement();
    pRoot2->SetAttribute("loadable",0);
    return STAT_OK;
}

int CaesarProject::loadProjectState(TiXmlNode* pRoot) {
    return STAT_OK;
}

int CaesarProject::createTestVectorFilesHeaders() const {
    ofstream tvFile;
    tvFile.open(FILE_TEST_VECTORS, ios_base::trunc | ios_base::binary);
    if (!tvFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    tvFile << dec << left;
    tvFile << pCaesarSettings->algorithm << " \t\t(algorithm:" << m_encryptor->shortDescription() << ")" << endl;
    tvFile << pCaesarSettings->limitAlgRounds << " \t\t(limit algorithm rounds?)" << endl;
    tvFile << pCaesarSettings->algorithmRoundsCount << " \t\t(algorithm rounds - if limited)" << endl;
    tvFile << pCaesarSettings->plaintextLength << " \t\t(plaintext length)" << endl;
    tvFile << pCaesarSettings->adLength << " \t\t(associated data length)" << endl;
    tvFile << pCaesarSettings->plaintextType << " \t\t(plaintext type)" << endl;
    tvFile << pCaesarSettings->keyType << " \t\t(key type)" << endl;
    tvFile << pCaesarSettings->adType << " \t\t(associated data type)" << endl;
    tvFile << pCaesarSettings->smnType << " \t\t(secret message number type)" << endl;
    tvFile << pCaesarSettings->pmnType << " \t\t(public message number type)" << endl;
    tvFile.close();

    return STAT_OK;
}

int CaesarProject::generateTestVectors() {
    int status = STAT_OK;

    // if set so, do not generate test vectors but generate data stream to cout
    if (pCaesarSettings->generateStream) {
        status = generateCipherDataStream();
        if (status != STAT_OK) {
            return status;
        } else {
            return STAT_INTENTIONAL_EXIT;
        }
    }

    switch (pCaesarSettings->usageType) {
    case CAESAR_DISTINGUISHER:
        // generate cipher stream
        for (int vector = 0; vector < pGlobals->settings->testVectors.setSize/2; vector++) {
            // ciphertext stream
            status = m_encryptor->encrypt(m_ciphertext, &m_realCiphertextLength);
            if (status != STAT_OK) { return status; }
            memcpy(pGlobals->testVectors.inputs[vector], m_ciphertext, pGlobals->settings->testVectors.inputLength);
            status = m_encryptor->update();
            if (status != STAT_OK) { return status; }
            // 0x00 to denote ciphertext stream
            for (int byte = 0; byte < pGlobals->settings->testVectors.outputLength; byte++) {
                pGlobals->testVectors.outputs[vector][byte] = 0;
            }
        }
        // generate random vectors
        for (int vector = pGlobals->settings->testVectors.setSize/2; vector < pGlobals->settings->testVectors.setSize; vector++) {
            // random stream
            for (int byte = 0; byte < pGlobals->settings->testVectors.inputLength; byte++) {
                rndGen->getRandomFromInterval(255, pGlobals->testVectors.inputs[vector] + byte);
            }
            // 0xff to denote random stream
            for (int byte = 0; byte < pGlobals->settings->testVectors.outputLength; byte++) {
                pGlobals->testVectors.outputs[vector][byte] = UCHAR_MAX;
            }
        }
        break;
    default:
        mainLogger.out(LOGGER_ERROR) << "unknown usage type (" << pCaesarSettings->usageType << ") in " << shortDescription() << endl;
        return STAT_INVALID_ARGUMETS;
        break;
    }

    return status;
}

int CaesarProject::generateCipherDataStream() {
    int status = STAT_OK;

    int st = m_encryptor->encrypt(m_ciphertext, &m_realCiphertextLength);
    mainLogger.out(LOGGER_INFO) << "Encryption status: " << st << endl;

    return status;
}
