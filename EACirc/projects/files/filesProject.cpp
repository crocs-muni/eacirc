#include "filesProject.h"

FILES_SETTINGS* pFileDistSettings = NULL;

FilesProject::FilesProject()
    : IProject(PROJECT_FILE_DISTINGUISHER), m_tvOutputs(NULL), m_tvInputs(NULL) {
    for (int i = 0; i < FILES_NUMBER_OF_FILES; i++) {
        m_readOffsets[i] = 0;
    }
    m_tvOutputs = new unsigned char[pGlobals->settings->testVectors.outputLength];
    memset(m_tvOutputs,0,pGlobals->settings->testVectors.outputLength);
    m_tvInputs = new unsigned char[pGlobals->settings->testVectors.inputLength];
    memset(m_tvInputs,0,pGlobals->settings->testVectors.inputLength);
    m_numVectors = new int[FILES_NUMBER_OF_FILES];
}

FilesProject::~FilesProject() {
    for (int fileNumber = 0; fileNumber < FILES_NUMBER_OF_FILES; fileNumber++) {
        if (m_files[fileNumber].is_open()) {
            m_files[fileNumber].close();
        }
    }
    if (m_tvOutputs != NULL) {
        delete[] m_tvOutputs;
        m_tvOutputs = NULL;
    }
    if (m_tvInputs != NULL) {
        delete[] m_tvInputs;
        m_tvInputs = NULL;
    }
    if (m_numVectors != NULL) {
        delete[] m_numVectors;
        m_numVectors = NULL;
    }
}

string FilesProject::shortDescription() const {
    return "file processor";
}

int FilesProject::loadProjectConfiguration(TiXmlNode* pRoot) {
    m_fileDistSettings.usageType = atoi(getXMLElementValue(pRoot,"FILE_DISTINGUISHER/USAGE_TYPE").c_str());
    m_fileDistSettings.filenames[0] = getXMLElementValue(pRoot,"FILE_DISTINGUISHER/FILENAME_1");
    m_fileDistSettings.filenames[1] = getXMLElementValue(pRoot,"FILE_DISTINGUISHER/FILENAME_2");
    m_fileDistSettings.ballancedTestVectors = atoi(getXMLElementValue(pRoot,"FILE_DISTINGUISHER/BALLANCED_TEST_VECTORS").c_str()) ? true : false;
    m_fileDistSettings.useFixedInitialOffset = atoi(getXMLElementValue(pRoot,"FILE_DISTINGUISHER/USE_FIXED_INITIAL_OFFSET").c_str()) ? true : false;
    istringstream ss;
    ss.str(getXMLElementValue(pRoot,"FILE_DISTINGUISHER/INITIAL_OFFSET_1"));
    ss >> m_fileDistSettings.initialOffsets[0];
    ss.str(getXMLElementValue(pRoot,"FILE_DISTINGUISHER/INITIAL_OFFSET_2"));
    ss >> m_fileDistSettings.initialOffsets[1];
    pFileDistSettings = &m_fileDistSettings;

    // bind project settings into global settings
    pGlobals->settings->project = (void*) pFileDistSettings;

    return STAT_OK;
}

int FilesProject::initializeProject() {
    for (int fileNumber = 0; fileNumber < FILES_NUMBER_OF_FILES; fileNumber++) {
        m_files[fileNumber].open(m_fileDistSettings.filenames[fileNumber], ios_base::binary);
        if (!m_files[fileNumber].is_open()) {
            mainLogger.out(LOGGER_ERROR) << "Could not open file (" << m_fileDistSettings.filenames[fileNumber] << ")." << endl;
            return STAT_FILE_OPEN_FAIL;
        }
        // determine file size
        m_files[fileNumber].seekg (0, ios::end);
        m_fileDistSettings.fileSizes[fileNumber] = m_files[fileNumber].tellg();
        m_files[fileNumber].seekg (0, ios::beg);
    }
    return STAT_OK;
}

int FilesProject::initializeProjectState() {
    if (!m_fileDistSettings.useFixedInitialOffset) {
        for (int i = 0; i < FILES_NUMBER_OF_FILES; i++) {
            rndGen->getRandomFromInterval(m_fileDistSettings.fileSizes[i],&(m_fileDistSettings.initialOffsets[i]));
        }
    }
    for (int i = 0; i < FILES_NUMBER_OF_FILES; i++) {
        m_readOffsets[i] = m_fileDistSettings.initialOffsets[i] % m_fileDistSettings.fileSizes[i];
        mainLogger.out(LOGGER_INFO) << "Using initial offset " << m_readOffsets[i] << " (" << m_fileDistSettings.filenames[i] << ")." << endl;
    }
    return STAT_OK;
}

int FilesProject::saveProjectState(TiXmlNode *pRoot) const {
    return STAT_OK;
}

int FilesProject::loadProjectState(TiXmlNode *pRoot) {
    return STAT_OK;
}

int FilesProject::createTestVectorFilesHeaders() const {
    // generate header (project config) to test vector file
    ofstream tvFile;
    tvFile.open(FILE_TEST_VECTORS, ios_base::app | ios_base::binary);
    if (!tvFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    tvFile << pGlobals->settings->main.projectType << " \t\t(project: " << shortDescription() << ")" << endl;
    tvFile << pFileDistSettings->usageType << " \t\t(usage type)" << endl;
    for (int i = 0; i < FILES_NUMBER_OF_FILES; i++) {
        tvFile << pFileDistSettings->filenames[i] << " \t\t(filename " << i << ")" << endl;
        tvFile << pFileDistSettings->initialOffsets[i] << " \t\t(initial offset for file " << i << ")" << endl;
    }
    tvFile << pFileDistSettings->ballancedTestVectors << " \t\t(ballanced test vectors?)" << endl;
    tvFile.close();

    // generate header to human-readable test-vector file
    tvFile.open(FILE_TEST_VECTORS_HR, ios::app | ios_base::binary);
    if (!tvFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    tvFile << "Using file contents (binary form) for test vector generation." << endl;
    for (int i = 0; i < FILES_NUMBER_OF_FILES; i++) {
        tvFile << "  file " << i << ": " << m_fileDistSettings.filenames[i] << endl;
        tvFile << "  initial reading offset: " << m_fileDistSettings.initialOffsets[i] << endl;
    }
    tvFile << "Test vectors formatted as INPUT::OUTPUT" << endl;
    tvFile << endl;
    tvFile.close();

    return STAT_OK;
}

int FilesProject::getStreamFromFile(int fileNumber, unsigned long length, unsigned char* data) {
    if (m_fileDistSettings.fileSizes[fileNumber]-m_readOffsets[fileNumber] < length) {
        mainLogger.out(LOGGER_WARNING) << "Not enought data in file, revinding (" << m_fileDistSettings.filenames[fileNumber] << ")." << endl;
        m_files[fileNumber].seekg(ios_base::beg);
        m_readOffsets[fileNumber] = m_files[fileNumber].tellg();
    }
    m_files[fileNumber].read((char*)data,length);
    if (m_files[fileNumber].fail()) {
        return STAT_FILE_READ_FAIL;
    } else {
        return STAT_OK;
    }
}

int FilesProject::prepareSingleTestVector() {
    int status = STAT_OK;
    //! are we using algorithm1 (0) or algorithm2 (1) ?
    int fileNumber = 0;
    switch (pFileDistSettings->usageType) {
    case FILES_DISTINGUISHER:
        //SHALL WE BALANCE TEST VECTORS?
        if (pFileDistSettings->ballancedTestVectors && (m_numVectors[0] >= pGlobals->settings->testVectors.setSize/2))
            fileNumber = 1;
        else if (pFileDistSettings->ballancedTestVectors && (m_numVectors[1] >= pGlobals->settings->testVectors.setSize/2))
            fileNumber = 0;
        else
            rndGen->getRandomFromInterval(1, &fileNumber);
        m_numVectors[fileNumber]++;
        // get correct input
        status = getStreamFromFile(fileNumber,pGlobals->settings->testVectors.inputLength,m_tvInputs);
        // set correct output
        for (int output = 0; output < pGlobals->settings->circuit.sizeOutputLayer; output++)
            m_tvOutputs[output] = fileNumber * 0xff;
        break;
    default:
        mainLogger.out(LOGGER_ERROR) << "Unknown usage type (" << pFileDistSettings->usageType << ")." << endl;
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

int FilesProject::generateTestVectors() {
    int status = STAT_OK;

    // USED FOR BALANCING TEST VECTORS
    for (int i = 0; i < FILES_NUMBER_OF_FILES; i++) {
        m_numVectors[i] = 0;
    }

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
