#include "fileDistinguisherProject.h"

FILE_DISTINGUISHER_SETTINGS* pFileDistSettings = NULL;

FileDistinguisherProject::FileDistinguisherProject()
    : IProject(PROJECT_FILE_DISTINGUISHER) {
    for (int i = 0; i < FILEDIST_NUMBER_OF_FILES; i++) {
        m_readOffsets[i] = 0;
    }
}

FileDistinguisherProject::~FileDistinguisherProject() {
    for (int fileNumber = 0; fileNumber < FILEDIST_NUMBER_OF_FILES; fileNumber++) {
        if (m_files[fileNumber].is_open()) {
            m_files[fileNumber].close();
        }
    }
}

string FileDistinguisherProject::shortDescription() const {
    return "file distinguisher";
}

int FileDistinguisherProject::loadProjectConfiguration(TiXmlNode* pRoot) {
    m_fileDistSettings.filenames[0] = getXMLElementValue(pRoot,"FILE_DISTINGUISHER/FILENAME_1");
    m_fileDistSettings.filenames[1] = getXMLElementValue(pRoot,"FILE_DISTINGUISHER/FILENAME_2");
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

int FileDistinguisherProject::initializeProject() {
    for (int fileNumber = 0; fileNumber < FILEDIST_NUMBER_OF_FILES; fileNumber++) {
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

int FileDistinguisherProject::initializeProjectState() {
    if (!m_fileDistSettings.useFixedInitialOffset) {
        for (int i = 0; i < FILEDIST_NUMBER_OF_FILES; i++) {
            rndGen->getRandomFromInterval(m_fileDistSettings.fileSizes[i],&(m_fileDistSettings.initialOffsets[i]));
        }
    }
    for (int i = 0; i < FILEDIST_NUMBER_OF_FILES; i++) {
        m_readOffsets[i] = m_fileDistSettings.initialOffsets[i] % m_fileDistSettings.fileSizes[i];
        mainLogger.out(LOGGER_INFO) << "Using initial offset " << m_readOffsets[i] << " (" << m_fileDistSettings.filenames[i] << ")." << endl;
    }
    return STAT_OK;
}

int FileDistinguisherProject::saveProjectState(TiXmlNode *pRoot) const {
    return STAT_NOT_IMPLEMENTED_YET;
}

int FileDistinguisherProject::loadProjectState(TiXmlNode *pRoot) {
    return STAT_NOT_IMPLEMENTED_YET;
}

int FileDistinguisherProject::createTestVectorFilesHeaders() const {
    return STAT_NOT_IMPLEMENTED_YET;
}

int FileDistinguisherProject::generateTestVectors() {
    return STAT_NOT_IMPLEMENTED_YET;
}
