//#include "stdafx.h"
#include "QuantumRndGen.h"
#include "time.h"

QuantumRndGen::QuantumRndGen(unsigned long seed, string QRBGSPath) 
		: m_usesQRNGData(false), m_fileIndex(0) {
    int status;
    m_accumulator = NULL;
    m_type = GENERATOR_QRNG;
    this->m_QRNGDataPath = QRBGSPath;

    // IF SEED NOT EXTERNALLY SUPPLIED, TAKE SYSTEM TIME
    if (seed == 0) {
        seed = (unsigned int) time(NULL);
        mainLogger.out() << "Warning: using system time to initialize random generator (" << this->shortDescription() << ")" << endl;
    }
    this->m_seed = seed;

    // INITIALIZE INTERNAL SYSTEM GENERATOR
    m_internalRNG.seed(seed);

    // CHECK FOR QUANTUM DATA SOURCE
    int length;
    ifstream file;
    ostringstream sFileName;
    sFileName << m_QRNGDataPath << FILE_QRNG_DATA_PREFIX << m_fileIndex << FILE_QRNG_DATA_SUFFIX;
    file.open(sFileName.str().c_str(), fstream::in | fstream::binary);
    if (file.is_open()) { // using QRNG data
        m_usesQRNGData = true;

        // DETERMINE DATA LENGTH
        file.seekg (0, ios::end);
        length = file.tellg();
        file.seekg (0, ios::beg);
        m_accLength = min(RANDOM_DATA_FILE_SIZE,length);

        file.close();
    } else { // QRNG data not available
        mainLogger.out() << "Warning: Quantum random data files not found, using system generator." << endl;
        m_accLength = 4; // (max 32B), see init and update
    }
    m_accumulator = new unsigned char[m_accLength];

    status = reinitRandomGenerator();

    if (status != STAT_OK) {
        mainLogger.out() << "Error: Random generator initialization failed. Subsequent program behavious undefined!" << endl;
        mainLogger.out() << "       status: " << ErrorToString(status) << endl;
    }
}

QuantumRndGen::~QuantumRndGen() {
    delete[] m_accumulator;
}


int QuantumRndGen::getRandomFromInterval(unsigned long highBound, unsigned long *pRandom) {
    int status = STAT_OK;

	if (highBound != ULONG_MAX) highBound++;
    // GET FIRST DWORD FROM ACCUMULATOR     
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

int QuantumRndGen::getRandomFromInterval(int highBound, int *pRandom) {
    int status = STAT_OK;

	if (highBound != INT_MAX) highBound++;
    // GET FIRST DWORD FROM ACCUMULATOR     
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
    // GET FIRST DWORD FROM ACCUMULATOR     
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
        m_accumulator[0] = m_internalRNG()%256;
        m_accumulator[1] = m_internalRNG()%256;
        m_accumulator[2] = m_internalRNG()%256;
        m_accumulator[3] = m_internalRNG()%256;
    } else {
        m_fileIndex = m_internalRNG() % FILE_QRNG_DATA_INDEX_MAX;
        status = loadQRNGDataFile();
        m_accPosition = m_internalRNG()%(m_accLength-4);
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
        m_accumulator[m_accPosition] = m_internalRNG()%256;
        m_accumulator[m_accPosition+1] = m_internalRNG()%256;
        m_accumulator[m_accPosition+2] = m_internalRNG()%256;
        m_accumulator[m_accPosition+3] = m_internalRNG()%256;
    }
	
    return status;
}

int QuantumRndGen::loadQRNGDataFile() {
    ifstream file;
    ostringstream sFileName;
    sFileName << m_QRNGDataPath << FILE_QRNG_DATA_PREFIX << m_fileIndex << FILE_QRNG_DATA_SUFFIX;
    file.open(sFileName.str().c_str(), fstream::in | fstream::binary);
    if (file.is_open()) {
        file.read((char*)m_accumulator, m_accLength);
        file.close();
    } else {
        mainLogger.out() << "Error: Cannot open QRNG data file " << sFileName.str() << "." << endl;
        return STAT_FILE_OPEN_FAIL;
    }
    return STAT_OK;
}

string QuantumRndGen::shortDescription() const {
    if (m_usesQRNGData) return "true quantum generator";
    else return "system-based random generator";
}

QuantumRndGen::QuantumRndGen(TiXmlNode* pRoot) {
    m_type = GENERATOR_QRNG;
    TiXmlElement* pElem = NULL;

    pElem = pRoot->FirstChildElement("original_seed");
    m_seed = atol(pElem->GetText());
    pElem = pRoot->FirstChildElement("qrng");
    m_usesQRNGData = atoi(pElem->Attribute("true_qrng")) == 1 ? true : false;
    if (m_usesQRNGData) {
        m_QRNGDataPath = pElem->FirstChildElement("data_path")->GetText();
        m_fileIndex = atoi(pElem->FirstChildElement("file_index")->GetText());
    } else {
        m_QRNGDataPath = "";
        m_fileIndex = 0;
    }
    pElem = pRoot->FirstChildElement("accumulator_state");
    m_accLength = atoi(pElem->Attribute("length"));
    m_accumulator = new unsigned char[m_accLength];
    m_accPosition = atoi(pElem->Attribute("position"));
    if (m_usesQRNGData) {
        loadQRNGDataFile();
    } else {
        istringstream ss(pElem->FirstChildElement("value")->GetText());
        unsigned int value;
        for (int i = 0; i < m_accLength; i++) {
            ss >> value;
            m_accumulator[i] = value;
        }
    }
    pElem = pRoot->FirstChildElement("internal_rng_state");
    istringstream ss(pElem->GetText());
    if (strcmp(pElem->Attribute("type"),typeid(m_internalRNG).name()) == 0) {
        ss >> m_internalRNG;
    } else {
        mainLogger.out() << "Error: Incompatible system generator type - state not loaded." << endl;
        mainLogger.out() << "       required: " << typeid(m_internalRNG).name() << endl;
        mainLogger.out() << "          found: " << pElem->Attribute("type") << endl;
    }
}

TiXmlNode* QuantumRndGen::exportGenerator() const {
    TiXmlElement* pRoot = new TiXmlElement("generator");
    pRoot->SetAttribute("type",shortDescription().c_str());

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

    TiXmlElement* internalRNGstate = new TiXmlElement("internal_rng_state");
    internalRNGstate->SetAttribute("type",typeid(m_internalRNG).name());
    stringstream state;
    state << dec << left << setfill(' ');
    state << m_internalRNG;
    internalRNGstate->LinkEndChild(new TiXmlText(state.str().c_str()));
    pRoot->LinkEndChild(internalRNGstate);

    return pRoot;
}
