#include "QuantumRndGen.h"
#include "time.h"
#include "MD5RndGen.h"
#include "EACglobals.h"
#include "XMLProcessor.h"

QuantumRndGen::QuantumRndGen(unsigned long seed, string QRBGSPath)
        : IRndGen(GENERATOR_QRNG,seed), m_usesQRNGData(false), m_fileIndex(0) {
    int status;
    m_accumulator = NULL;
    this->m_QRNGDataPath = QRBGSPath;

    // INITIALIZE INTERNAL GENERATOR
    m_internalRNG = new MD5RndGen(seed);

    checkQRNGdataAvailability();
    m_accumulator = new unsigned char[m_accLength];
    status = reinitRandomGenerator();

    if (status != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Random generator initialization failed. Subsequent program behavious undefined!" << endl;
        mainLogger.out() << "       status: " << statusToString(status) << endl;
    }
}

QuantumRndGen::~QuantumRndGen() {
    delete[] m_accumulator;
    delete m_internalRNG;
}

void QuantumRndGen::checkQRNGdataAvailability() {
    int length;
    size_t separatorPosition;
    string paths = m_QRNGDataPath;
    while (!paths.empty()) {
        string currentPath = paths.substr(0,paths.find(FILE_PATH_SEPARATOR));
        if ((separatorPosition = paths.find(FILE_PATH_SEPARATOR)) == paths.npos) {
            paths = "";
        } else {
            paths = paths.substr(separatorPosition+1,paths.npos-separatorPosition-1);
        }
        m_QRNGDataPath = currentPath;
        ifstream file;
        file.open(getQRNGDataFileName(m_fileIndex).c_str(), fstream::in | fstream::binary);
        if (file.is_open()) { // using QRNG data
            m_usesQRNGData = true;
            // DETERMINE DATA LENGTH
            file.seekg (0, ios::end);
            length = file.tellg();
            file.seekg (0, ios::beg);
            m_accLength = min(RANDOM_DATA_FILE_SIZE,length);
            file.close();
            mainLogger.out(LOGGER_INFO) << "Quantum random data found at \"" << currentPath << "\"." << endl;
            return;
        }
    }

    // QRNG data not available
    m_QRNGDataPath = "";
    mainLogger.out(LOGGER_WARNING) << "Quantum random data files not found (" << getQRNGDataFileName(m_fileIndex) << "), using system generator." << endl;
    m_accLength = 4; // (max 32B), see init and update
}

string QuantumRndGen::getQRNGDataFileName(int fileIndex) {
    ostringstream sFileName;
    int indexWidth;
    if (pGlobals->settings->random.qrngFilesMaxIndex == 0) {
        indexWidth = 1;
    } else {
        indexWidth =  floor(log10(pGlobals->settings->random.qrngFilesMaxIndex)) + 1;
    }
    sFileName << m_QRNGDataPath << FILE_QRNG_DATA_PREFIX;
    sFileName << dec << right << setw(indexWidth) << setfill('0') << fileIndex;
    sFileName << FILE_QRNG_DATA_SUFFIX;

    return sFileName.str();
}

int QuantumRndGen::getRandomFromInterval(unsigned long highBound, unsigned long *pRandom) {
    int status = STAT_OK;

    if (highBound != ULONG_MAX) highBound++;
    // GET FIRST ULONG FROM ACCUMULATOR
    unsigned long   random;
    memcpy(&random, m_accumulator+m_accPosition, sizeof(unsigned long));
    if (pRandom) {
        *pRandom = (unsigned long) (((float) random / ULONG_MAX) *  highBound);
		if (*pRandom == highBound) *pRandom = 0;
	}
	// UPDATE ACCUMULATOR
    status = updateAccumulator();

    return status;
}

int QuantumRndGen::getRandomFromInterval(unsigned char highBound, unsigned char *pRandom) {
    int status = STAT_OK;
    unsigned long   rand = 0;
    
    status = getRandomFromInterval(highBound, &rand);
    *pRandom = (unsigned char) rand;

    return status;
}

int QuantumRndGen::getRandomFromInterval(unsigned int highBound, unsigned int *pRandom) {
    int status = STAT_OK;
    unsigned long   rand = 0;

    status = getRandomFromInterval(highBound, &rand);
    *pRandom = (unsigned int) rand;

    return status;
}

int QuantumRndGen::getRandomFromInterval(int highBound, int *pRandom) {
    int status = STAT_OK;

	if (highBound != INT_MAX) highBound++;
    // GET FIRST ULONG FROM ACCUMULATOR
    int   random;
    memcpy(&random, m_accumulator+m_accPosition, sizeof(int));
    // SUPRESS NEGATIVE VALUES
    random = abs(random);
    if (pRandom) {
		*pRandom = (int) (((float) random / INT_MAX) *  highBound);
		if (*pRandom == highBound) *pRandom = 0;
	}

	// UPDATE ACCUMULATOR
    status = updateAccumulator();

    return status;
}

int QuantumRndGen::getRandomFromInterval(float highBound, float *pRandom) {
    int status = STAT_OK;

    if (highBound != ULONG_MAX) highBound++;
    // GET FIRST ULONG FROM ACCUMULATOR
    unsigned long   random;
    memcpy(&random, m_accumulator+m_accPosition, sizeof(unsigned long));
    if (pRandom) {
        *pRandom = (float) (((float) random / ULONG_MAX) *  highBound);
		if (*pRandom == highBound) *pRandom = 0;
	}

	// UPDATE ACCUMULATOR
    status = updateAccumulator();

    return status;
}

int QuantumRndGen::discartValue() {
    return updateAccumulator();
}

int QuantumRndGen::reinitRandomGenerator() {
    int status = STAT_OK;
    if (!m_usesQRNGData) {
        m_accPosition = 0;
        // MAX 32B INT = 256x256x256x256
        m_internalRNG->getRandomFromInterval(UCHAR_MAX,m_accumulator + 0);
        m_internalRNG->getRandomFromInterval(UCHAR_MAX,m_accumulator + 1);
        m_internalRNG->getRandomFromInterval(UCHAR_MAX,m_accumulator + 2);
        m_internalRNG->getRandomFromInterval(UCHAR_MAX,m_accumulator + 3);
    } else {
        do {
            m_internalRNG->getRandomFromInterval(pGlobals->settings->random.qrngFilesMaxIndex,&m_fileIndex);
            status = loadQRNGDataFile();
        } while (status != STAT_OK);
        m_internalRNG->getRandomFromInterval(m_accLength-4, &m_accPosition);
    }
    return status;
}

int QuantumRndGen::updateAccumulator() {
    int status = STAT_OK;
    if (m_usesQRNGData) {
        // using QRNG data files
        m_accPosition += 4;
        if ((m_accPosition+4) > m_accLength)
            status = reinitRandomGenerator();
    } else {
        // using system generator (accPosition = 0)
        m_internalRNG->getRandomFromInterval(UCHAR_MAX,m_accumulator + m_accPosition + 0);
        m_internalRNG->getRandomFromInterval(UCHAR_MAX,m_accumulator + m_accPosition + 1);
        m_internalRNG->getRandomFromInterval(UCHAR_MAX,m_accumulator + m_accPosition + 2);
        m_internalRNG->getRandomFromInterval(UCHAR_MAX,m_accumulator + m_accPosition + 3);
    }
	
    return status;
}

int QuantumRndGen::loadQRNGDataFile() {
    ifstream file;
    file.open(getQRNGDataFileName(m_fileIndex).c_str(), fstream::in | fstream::binary);
    if (file.is_open()) {
        file.read((char*)m_accumulator, m_accLength);
        file.close();
    } else {
        mainLogger.out(LOGGER_WARNING) << "Cannot open QRNG data file " << getQRNGDataFileName(m_fileIndex) << "." << endl;
        return STAT_FILE_OPEN_FAIL;
    }
    return STAT_OK;
}

string QuantumRndGen::shortDescription() const {
    if (m_usesQRNGData) return "true quantum generator";
    else return "system-based random generator";
}

QuantumRndGen::QuantumRndGen(TiXmlNode *pRoot)
        : IRndGen(GENERATOR_QRNG,1), // cannot call IRndGen with seed 0, warning would be issued
          m_accumulator(NULL), m_usesQRNGData(false), m_accLength(0), m_accPosition(0), m_fileIndex(0), m_internalRNG(NULL) {
    if (atoi(getXMLElementValue(pRoot,"@type").c_str()) != m_type) {
        mainLogger.out(LOGGER_ERROR) << "Incompatible generator types." << endl;
        return;
    }

    istringstream(getXMLElementValue(pRoot,"original_seed")) >> m_seed;
    m_usesQRNGData = atoi(getXMLElementValue(pRoot,"qrng/@true_qrng").c_str()) == 1 ? true : false;
    if (m_usesQRNGData) {
        m_QRNGDataPath = getXMLElementValue(pRoot,"qrng/data_path");
        m_fileIndex = atoi(getXMLElementValue(pRoot,"qrng/file_index").c_str());
    } else {
        m_QRNGDataPath = "";
        m_fileIndex = 0;
    }
    m_accLength = atoi(getXMLElementValue(pRoot,"accumulator_state/@length").c_str());
    m_accumulator = new unsigned char[m_accLength];
    m_accPosition = atoi(getXMLElementValue(pRoot,"accumulator_state/@position").c_str());
    if (m_usesQRNGData) {
        loadQRNGDataFile();
    } else {
        istringstream ss(getXMLElementValue(pRoot,"accumulator_state/value"));
        unsigned int value;
        for (int i = 0; i < m_accLength; i++) {
            ss >> value;
            m_accumulator[i] = value;
        }
    }
    m_internalRNG = new MD5RndGen(getXMLElement(pRoot,"internal_rng/generator"));
}

TiXmlNode* QuantumRndGen::exportGenerator() const {
    TiXmlElement* pRoot = new TiXmlElement("generator");
    pRoot->SetAttribute("type",toString(m_type).c_str());
    pRoot->SetAttribute("description",shortDescription().c_str());

    TiXmlElement* originalSeed = new TiXmlElement("original_seed");
    stringstream sSeed;
    sSeed << m_seed;
    originalSeed->LinkEndChild(new TiXmlText(sSeed.str().c_str()));
    pRoot->LinkEndChild(originalSeed);

    TiXmlElement* qrng = new TiXmlElement("qrng");
    qrng->SetAttribute("true_qrng",m_usesQRNGData ? "1" : "0");
    TiXmlElement* QRNGpath = new TiXmlElement("data_path");
    QRNGpath->LinkEndChild(new TiXmlText(m_QRNGDataPath.c_str()));
    qrng->LinkEndChild(QRNGpath);
    TiXmlElement* fileIndex = new TiXmlElement("file_index");
    stringstream sFileIndex;
    sFileIndex << m_fileIndex;
    fileIndex->LinkEndChild(new TiXmlText(sFileIndex.str().c_str()));
    qrng->LinkEndChild(fileIndex);
    pRoot->LinkEndChild(qrng);

    TiXmlElement* accumulatorState = new TiXmlElement("accumulator_state");
    accumulatorState->SetAttribute("length",m_accLength);
    accumulatorState->SetAttribute("position",m_accPosition);
    TiXmlElement* value = new TiXmlElement("value");
    if (!m_usesQRNGData) {
        stringstream sAccValue;
        sAccValue << left << dec;
        sAccValue << (int)m_accumulator[0] << " ";
        sAccValue << (int)m_accumulator[1] << " ";
        sAccValue << (int)m_accumulator[2] << " ";
        sAccValue << (int)m_accumulator[3];
        value->LinkEndChild(new TiXmlText(sAccValue.str().c_str()));
    }
    accumulatorState->LinkEndChild(value);
    pRoot->LinkEndChild(accumulatorState);

    TiXmlElement* internalRNGstate = new TiXmlElement("internal_rng");
    internalRNGstate->LinkEndChild(m_internalRNG->exportGenerator());
    pRoot->LinkEndChild(internalRNGstate);

    return pRoot;
}
