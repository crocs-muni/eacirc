#include "filesProject.h"

FILES_SETTINGS* pFilesSettings = NULL;

FilesProject::FilesProject()
    : IProject(PROJECT_FILE_DISTINGUISHER), m_tvOutputs(NULL), m_tvInputs(NULL) {
    for (int i = 0; i < FILES_NUMBER_OF_FILES; i++) {
        m_readOffsets[i] = 0;
        m_rewinds[i] = 0;
    }
    m_tvOutputs = new unsigned char[pGlobals->settings->testVectors.outputLength];
    memset(m_tvOutputs,0,pGlobals->settings->testVectors.outputLength);
    m_tvInputs = new unsigned char[pGlobals->settings->testVectors.inputLength];
    memset(m_tvInputs,0,pGlobals->settings->testVectors.inputLength);
    m_numVectors = new int[FILES_NUMBER_OF_FILES];
}

FilesProject::~FilesProject() {
    mainLogger.out(LOGGER_INFO) << "File rewinding summary:" << endl;
    for (int file = 0; file < FILES_NUMBER_OF_FILES; file++) {
        if (m_rewinds[file] == 0) {
            mainLogger.out(LOGGER_INFO) << "File stream " << file+1 << " was not rewound (" << m_filesSettings.filenames[file] << ")." << endl;
        } else {
            mainLogger.out(LOGGER_WARNING) << "File stream " << file+1 << " was rewound " << m_rewinds[file];
            mainLogger.out() << " times (" << m_filesSettings.filenames[file] << ")." << endl;
        }
    }

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

string FilesProject::testingConfiguration() {
    string config =
            "<FILES>"
            "    <USAGE_TYPE>401</USAGE_TYPE>"
            "    <FILENAME_1>../../qrng/Random000.bin</FILENAME_1>"
            "    <FILENAME_2>../../qrng/Random001.bin</FILENAME_2>"
            "    <BALLANCED_TEST_VECTORS>1</BALLANCED_TEST_VECTORS>"
            "    <USE_FIXED_INITIAL_OFFSET>1</USE_FIXED_INITIAL_OFFSET>"
            "    <INITIAL_OFFSET_1>1048576</INITIAL_OFFSET_1>"
            "    <INITIAL_OFFSET_2>1048000</INITIAL_OFFSET_2>"
            "</FILES>";
    return config;
}

int FilesProject::loadProjectConfiguration(TiXmlNode* pRoot) {
    m_filesSettings.usageType = atoi(getXMLElementValue(pRoot,"FILES/USAGE_TYPE").c_str());
    m_filesSettings.filenames[0] = getXMLElementValue(pRoot,"FILES/FILENAME_1");
    m_filesSettings.filenames[1] = getXMLElementValue(pRoot,"FILES/FILENAME_2");
    m_filesSettings.ballancedTestVectors = atoi(getXMLElementValue(pRoot,"FILES/BALLANCED_TEST_VECTORS").c_str()) ? true : false;
    m_filesSettings.useFixedInitialOffset = atoi(getXMLElementValue(pRoot,"FILES/USE_FIXED_INITIAL_OFFSET").c_str()) ? true : false;
    istringstream ss;
    ss.str(getXMLElementValue(pRoot,"FILES/INITIAL_OFFSET_1"));
    ss >> m_filesSettings.initialOffsets[0];
    ss.clear();
    ss.str(getXMLElementValue(pRoot,"FILES/INITIAL_OFFSET_2"));
    ss >> m_filesSettings.initialOffsets[1];
    pFilesSettings = &m_filesSettings;

    // bind project settings into global settings
    pGlobals->settings->project = (void*) pFilesSettings;

    return STAT_OK;
}

int FilesProject::initializeProject() {
    for (int fileNumber = 0; fileNumber < FILES_NUMBER_OF_FILES; fileNumber++) {
        m_files[fileNumber].open(m_filesSettings.filenames[fileNumber], ios_base::binary);
        if (!m_files[fileNumber].is_open()) {
            mainLogger.out(LOGGER_ERROR) << "Could not open file (" << m_filesSettings.filenames[fileNumber] << ")." << endl;
            return STAT_FILE_OPEN_FAIL;
        }
        // determine file size
        m_files[fileNumber].seekg (0, ios::end);
        m_filesSettings.fileSizes[fileNumber] = m_files[fileNumber].tellg();
        m_files[fileNumber].seekg (0, ios::beg);
    }
    return STAT_OK;
}

int FilesProject::initializeProjectState() {
    if (!m_filesSettings.useFixedInitialOffset) {
        for (int i = 0; i < FILES_NUMBER_OF_FILES; i++) {
            rndGen->getRandomFromInterval(m_filesSettings.fileSizes[i],&(m_filesSettings.initialOffsets[i]));
        }
    }
    for (int i = 0; i < FILES_NUMBER_OF_FILES; i++) {
        m_readOffsets[i] = m_filesSettings.initialOffsets[i] % m_filesSettings.fileSizes[i];
        m_files[i].seekg(m_readOffsets[i]);
        mainLogger.out(LOGGER_INFO) << "Using initial offset " << m_readOffsets[i] << " (" << m_filesSettings.filenames[i] << ")." << endl;
    }
    return STAT_OK;
}

int FilesProject::saveProjectState(TiXmlNode *pRoot) const {
    TiXmlElement* pRoot2 = pRoot->ToElement();
    TiXmlElement* pElem = NULL;
    TiXmlElement* pElem2 = NULL;
    pRoot2->SetAttribute("loadable",1);
    pElem = new TiXmlElement("usage_type");
    pElem->LinkEndChild(new TiXmlText(CommonFnc::toString(m_filesSettings.usageType).c_str()));
    pRoot2->LinkEndChild(pElem);
    for (int file = 0; file < FILES_NUMBER_OF_FILES; file++) {
        pElem = new TiXmlElement((string("file_")+CommonFnc::toString(file+1)).c_str());
        pElem2 = new TiXmlElement("filename");
        pElem2->LinkEndChild(new TiXmlText(m_filesSettings.filenames[file].c_str()));
        pElem->LinkEndChild(pElem2);
        pElem2 = new TiXmlElement("file_size");
        pElem2->LinkEndChild(new TiXmlText(CommonFnc::toString(m_filesSettings.fileSizes[file]).c_str()));
        pElem->LinkEndChild(pElem2);
        pElem2 = new TiXmlElement("initial_read_offset");
        pElem2->LinkEndChild(new TiXmlText(CommonFnc::toString(m_filesSettings.initialOffsets[file]).c_str()));
        pElem->LinkEndChild(pElem2);
        pElem2 = new TiXmlElement("current_read_offset");
        pElem2->LinkEndChild(new TiXmlText(CommonFnc::toString(m_readOffsets[file]).c_str()));
        pElem->LinkEndChild(pElem2);
        pRoot2->LinkEndChild(pElem);
    }
    return STAT_OK;
}

int FilesProject::loadProjectState(TiXmlNode *pRoot) {
    int status = STAT_OK;
    if (atoi(getXMLElementValue(pRoot,"usage_type").c_str()) != m_filesSettings.usageType) {
        mainLogger.out(LOGGER_ERROR) << "Incompatible usage types." << endl;
        return STAT_CONFIG_INCORRECT;
    }
    string configPrefix;
    istringstream ss;
    unsigned long value;
    for (int file = 0; file < FILES_NUMBER_OF_FILES; file++) {
        configPrefix = "file_" + CommonFnc::toString(file+1);
        ss.str(getXMLElementValue(pRoot,configPrefix+"/file_size"));
        ss >> value;
        if (value != m_filesSettings.fileSizes[file]) {
            mainLogger.out(LOGGER_WARNING) << "Different file size from previous run (" << m_filesSettings.filenames[file] << ")." << endl;
        }
        ss.clear();
        ss.str(getXMLElementValue(pRoot,configPrefix+"/current_read_offset"));
        ss >> m_readOffsets[file];
        ss.clear();
        ss.str(getXMLElementValue(pRoot,configPrefix+"/initial_read_offset"));
        ss >> m_filesSettings.initialOffsets[file];

        m_files[file].seekg(m_readOffsets[file]);
        mainLogger.out(LOGGER_INFO) << "Using initial offset " << m_readOffsets[file] << " (" << m_filesSettings.filenames[file] << ")." << endl;
    }

    return status;
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
    tvFile << pFilesSettings->usageType << " \t\t(usage type)" << endl;
    for (int i = 0; i < FILES_NUMBER_OF_FILES; i++) {
        tvFile << pFilesSettings->filenames[i] << " \t\t(filename " << i << ")" << endl;
        tvFile << pFilesSettings->initialOffsets[i] << " \t\t(initial offset for file " << i << ")" << endl;
    }
    tvFile << pFilesSettings->ballancedTestVectors << " \t\t(ballanced test vectors?)" << endl;
    tvFile.close();

    // generate header to human-readable test-vector file
    tvFile.open(FILE_TEST_VECTORS_HR, ios::app | ios_base::binary);
    if (!tvFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    tvFile << "Using file contents (binary form) for test vector generation." << endl;
    for (int i = 0; i < FILES_NUMBER_OF_FILES; i++) {
        tvFile << "  file " << i << ": " << m_filesSettings.filenames[i] << endl;
        tvFile << "  initial reading offset: " << m_filesSettings.initialOffsets[i] << endl;
    }
    tvFile << "Test vectors formatted as INPUT::OUTPUT" << endl;
    tvFile << endl;
    tvFile.close();

    return STAT_OK;
}

int FilesProject::getStreamFromFile(int fileNumber, unsigned long length, unsigned char* data) {
    if (m_filesSettings.fileSizes[fileNumber]-m_readOffsets[fileNumber] < length) {
        mainLogger.out(LOGGER_WARNING) << "Not enought data in file, rewinding (" << m_filesSettings.filenames[fileNumber] << ")." << endl;
        m_files[fileNumber].seekg(ios_base::beg);
        m_rewinds[fileNumber]++;
        m_readOffsets[fileNumber] = m_files[fileNumber].tellg();
    }
    m_files[fileNumber].read((char*)data,length);
    m_readOffsets[fileNumber] += m_files[fileNumber].gcount();
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
    switch (pFilesSettings->usageType) {
    case FILES_DISTINGUISHER:
        //SHALL WE BALANCE TEST VECTORS?
        if (pFilesSettings->ballancedTestVectors && (m_numVectors[0] >= pGlobals->settings->testVectors.setSize/2))
            fileNumber = 1;
        else if (pFilesSettings->ballancedTestVectors && (m_numVectors[1] >= pGlobals->settings->testVectors.setSize/2))
            fileNumber = 0;
        else
            rndGen->getRandomFromInterval(1, &fileNumber);
        m_numVectors[fileNumber]++;
        // get correct input
        status = getStreamFromFile(fileNumber,pGlobals->settings->testVectors.inputLength,m_tvInputs);
        // set correct output
        for (int output = 0; output < pGlobals->settings->main.circuitSizeOutput; output++)
            m_tvOutputs[output] = fileNumber * 0xff;
        break;
    default:
        mainLogger.out(LOGGER_ERROR) << "Unknown usage type (" << pFilesSettings->usageType << ")." << endl;
        return STAT_INVALID_ARGUMETS;
        break;
    }

    // save human-readable test vector
    if (pGlobals->settings->outputs.saveTestVectors) {
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
        if (pGlobals->settings->outputs.saveTestVectors == 1) {
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
