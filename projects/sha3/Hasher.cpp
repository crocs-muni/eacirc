#include "Hasher.h"
#include "hash_functions/hashFunctions.h"
#include "generators/IRndGen.h"

Hasher::Hasher() : m_initSuccess(false) {
    int algorithm = -1;
    int numRounds = -1;

    // initialize hash function arrays
    for (int algorithmNumber = 0; algorithmNumber < 2; algorithmNumber++) {
        m_hashFunctions[algorithmNumber] = NULL;
        m_hashOutputs[algorithmNumber] = NULL;
    }

    for (int algorithmNumber = 0; algorithmNumber < 2; algorithmNumber++) {
        // get correct settings for this hash function
        switch (algorithmNumber) {
        case 0:
            algorithm = pSha3Settings->algorithm1;
            numRounds = pSha3Settings->limitAlgRounds ? pSha3Settings->alg1RoundsCount : -1;
            m_hashOutputLengths[algorithmNumber] = pSha3Settings->hashLength1/8;
            break;
        case 1:
            algorithm = pSha3Settings->algorithm2;
            numRounds = pSha3Settings->limitAlgRounds ? pSha3Settings->alg1RoundsCount : -1;
            m_hashOutputLengths[algorithmNumber] = pSha3Settings->hashLength2/8;
            break;
        default:
            mainLogger.out(LOGGER_ERROR) << "Unsupported Hasher iteration in initialization (";
            mainLogger.out() << algorithmNumber << ")." << endl;
        }
        // allocate hash functions
        m_hashFunctions[algorithmNumber] = NULL;
        m_hashOutputs[algorithmNumber] = NULL;
        m_usedBytes[algorithmNumber] = m_hashOutputLengths[algorithmNumber];
        m_counters[algorithmNumber] = 0;
        if (algorithm == SHA3_RANDOM) continue;

        // if algorithm is set (other than random), allocate hash function
        m_hashOutputs[algorithmNumber] = new unsigned char[m_hashOutputLengths[algorithmNumber]];
        // check hassOutputLength-testVectorInputSize relation
        if (m_hashOutputLengths[algorithmNumber] % pGlobals->settings->testVectors.inputLength != 0) {
            mainLogger.out(LOGGER_WARNING) << "Test vector input length is not divider of hash output length." << endl;
            mainLogger.out(LOGGER_WARNING) << "Some bytes of hash will not be used (not unrecommended!)." << endl;
        }
        switch (algorithm) {
        case SHA3_ABACUS:       m_hashFunctions[algorithmNumber] = new Abacus(numRounds); break;
        case SHA3_ARIRANG:      m_hashFunctions[algorithmNumber] = new Arirang(numRounds); break;
        case SHA3_AURORA:       m_hashFunctions[algorithmNumber] = new Aurora(numRounds); break;
        case SHA3_BLAKE:        m_hashFunctions[algorithmNumber] = new Blake(numRounds); break;
        case SHA3_BLENDER:      m_hashFunctions[algorithmNumber] = new Blender(numRounds); break;
        case SHA3_BMW:          m_hashFunctions[algorithmNumber] = new BMW(numRounds); break;
        case SHA3_BOOLE:        m_hashFunctions[algorithmNumber] = new Boole(numRounds); break;
        case SHA3_CHEETAH:      m_hashFunctions[algorithmNumber] = new Cheetah(numRounds); break;
        case SHA3_CHI:          m_hashFunctions[algorithmNumber] = new Chi(numRounds); break;
        case SHA3_CRUNCH:       m_hashFunctions[algorithmNumber] = new Crunch(numRounds); break;
        case SHA3_CUBEHASH:     m_hashFunctions[algorithmNumber] = new Cubehash(numRounds); break;
        case SHA3_DCH:          m_hashFunctions[algorithmNumber] = new DCH(numRounds); break;
        case SHA3_DYNAMICSHA:   m_hashFunctions[algorithmNumber] = new DSHA(numRounds); break;
        case SHA3_DYNAMICSHA2:  m_hashFunctions[algorithmNumber] = new DSHA2(numRounds); break;
        case SHA3_ECHO:         m_hashFunctions[algorithmNumber] = new Echo(numRounds); break;
            //case SHA3_ECOH:         m_hashFunctions[algorithmNumber] = new Ecoh(numRounds); break;
        case SHA3_EDON:         checkNumRounds(numRounds,SHA3_EDON);
                                m_hashFunctions[algorithmNumber] = new Edon; break;
            //case SHA3_ENRUPT:       m_hashFunctions[algorithmNumber] = new EnRUPT(numRounds); break;
        case SHA3_ESSENCE:      m_hashFunctions[algorithmNumber] = new Essence(numRounds); break;
        case SHA3_FUGUE:        m_hashFunctions[algorithmNumber] = new Fugue(numRounds); break;
        case SHA3_GROSTL:       m_hashFunctions[algorithmNumber] = new Grostl(numRounds); break;
        case SHA3_HAMSI:        m_hashFunctions[algorithmNumber] = new Hamsi(numRounds); break;
        case SHA3_JH:           m_hashFunctions[algorithmNumber] = new JH(numRounds); break;
        case SHA3_KECCAK:       checkNumRounds(numRounds,SHA3_KECCAK);
                                m_hashFunctions[algorithmNumber] = new Keccak; break;
        case SHA3_KHICHIDI:     checkNumRounds(numRounds,SHA3_KHICHIDI);
                                m_hashFunctions[algorithmNumber] = new Khichidi; break;
        case SHA3_LANE:         m_hashFunctions[algorithmNumber] = new Lane(numRounds); break;
        case SHA3_LESAMNTA:     m_hashFunctions[algorithmNumber] = new Lesamnta(numRounds); break;
        case SHA3_LUFFA:        m_hashFunctions[algorithmNumber] = new Luffa(numRounds); break;
            //case SHA3_LUX:          m_hashFunctions[algorithmNumber] = new Lux(numRounds); break;
        case SHA3_MSCSHA3:      checkNumRounds(numRounds,SHA3_MSCSHA3);
                                m_hashFunctions[algorithmNumber] = new Mscsha; break;
        case SHA3_MD6:          m_hashFunctions[algorithmNumber] = new MD6(numRounds); break;
        case SHA3_MESHHASH:     m_hashFunctions[algorithmNumber] = new MeshHash(numRounds); break;
        case SHA3_NASHA:        checkNumRounds(numRounds,SHA3_NASHA);
                                m_hashFunctions[algorithmNumber] = new Nasha; break;
            //case SHA3_SANDSTORM:    m_hashFunctions[algorithmNumber] = new Sandstorm(numRounds); break;
        case SHA3_SARMAL:       m_hashFunctions[algorithmNumber] = new Sarmal(numRounds); break;
        case SHA3_SHABAL:       checkNumRounds(numRounds,SHA3_SARMAL);
                                m_hashFunctions[algorithmNumber] = new Shabal; break;
        case SHA3_SHAMATA:      checkNumRounds(numRounds,SHA3_SHAMATA);
                                m_hashFunctions[algorithmNumber] = new Shamata; break;
        case SHA3_SHAVITE3:     m_hashFunctions[algorithmNumber] = new SHAvite(numRounds); break;
        case SHA3_SIMD:         m_hashFunctions[algorithmNumber] = new Simd(numRounds); break;
        case SHA3_SKEIN:        checkNumRounds(numRounds,SHA3_SKEIN);
                                m_hashFunctions[algorithmNumber] = new Skein; break;
        case SHA3_SPECTRALHASH: checkNumRounds(numRounds,SHA3_SPECTRALHASH);
                                m_hashFunctions[algorithmNumber] = new SpectralHash; break;
        case SHA3_STREAMHASH:   checkNumRounds(numRounds,SHA3_STREAMHASH);
                                m_hashFunctions[algorithmNumber] = new StreamHash; break;
            //case SHA3_SWIFFTX:      m_hashFunctions[algorithmNumber] = new Swifftx(numRounds); break;
        case SHA3_TANGLE:       m_hashFunctions[algorithmNumber] = new Tangle(numRounds); break;
            //case SHA3_TIB3:         m_hashFunctions[algorithmNumber] = new Tib3(numRounds); break;
        case SHA3_TWISTER:      m_hashFunctions[algorithmNumber] = new Twister(numRounds); break;
        case SHA3_VORTEX:       m_hashFunctions[algorithmNumber] = new Vortex(numRounds); break;
        case SHA3_WAMM:         m_hashFunctions[algorithmNumber] = new WaMM(numRounds); break;
        case SHA3_WATERFALL:    m_hashFunctions[algorithmNumber] = new Waterfall(numRounds); break;
        case SHA3_TANGLE2:      m_hashFunctions[algorithmNumber] = new Tangle2(numRounds); break;
        default:
            mainLogger.out(LOGGER_ERROR) << "Unknown hash function type (" << algorithm << ")." << endl;
            return;
        }
    }
    m_initSuccess = true;
}

bool Hasher::initSuccess() const {
    return m_initSuccess;
}

Hasher::~Hasher() {
    for (int algorithmNumber = 0; algorithmNumber < 2; algorithmNumber++) {
        if (m_hashFunctions[algorithmNumber] != NULL)
            delete m_hashFunctions[algorithmNumber];
        m_hashFunctions[algorithmNumber] = NULL;
        if (m_hashOutputs[algorithmNumber] != NULL)
            delete[] m_hashOutputs[algorithmNumber];
        m_hashOutputs[algorithmNumber] = NULL;
    }
}

int Hasher::initializeState() {
    if (!pSha3Settings->useFixedSeed) {
        rndGen->getRandomFromInterval(ULONG_MAX,&(pSha3Settings->seed));
    }
    for (int algorithmNumber = 0; algorithmNumber < 2; algorithmNumber++) {
        m_counters[algorithmNumber] = pSha3Settings->seed;
    }
    return STAT_OK;
}

int Hasher::saveHasherState(TiXmlNode* pRoot) const {
    TiXmlElement* pElem = NULL;
    TiXmlElement* pElem2 = NULL;
    for (int algorithmNumber = 0; algorithmNumber < 2; algorithmNumber++) {
        int algorithm = algorithmNumber==0 ? pSha3Settings->algorithm1 : pSha3Settings->algorithm2;
        pElem = new TiXmlElement((string("algorithm")+CommonFnc::toString(algorithmNumber+1)).c_str());
        pElem->SetAttribute("type",algorithm);
        pElem->SetAttribute("description",Sha3Functions::sha3ToString(algorithm));
        if (algorithm != SHA3_RANDOM) {
            pElem2 = new TiXmlElement("hash_output_length");
            pElem2->LinkEndChild(new TiXmlText(CommonFnc::toString(m_hashOutputLengths[algorithmNumber]).c_str()));
            pElem->LinkEndChild(pElem2);
            pElem2 = new TiXmlElement("counter");
            pElem2->LinkEndChild(new TiXmlText(CommonFnc::toString(m_counters[algorithmNumber]).c_str()));
            pElem->LinkEndChild(pElem2);
            pElem2 = new TiXmlElement("used_bytes");
            pElem2->LinkEndChild(new TiXmlText(CommonFnc::toString(m_usedBytes[algorithmNumber]).c_str()));
            pElem->LinkEndChild(pElem2);
            ostringstream hash;
            for (int byte = 0; byte < m_hashOutputLengths[algorithmNumber]; byte++) {
                hash << dec << (int)m_hashOutputs[algorithmNumber][byte] << " ";
            }
            pElem2 = new TiXmlElement("current_hash");
            pElem2->LinkEndChild(new TiXmlText(hash.str().c_str()));
            pElem->LinkEndChild(pElem2);
        }
        pRoot->LinkEndChild(pElem);
    }
    return STAT_OK;
}

int Hasher::loadHasherState(TiXmlNode* pRoot) {
    int status = STAT_OK;
    TiXmlNode* pElem = NULL;
    int algorithm;
    for (int algorithmNumber = 0; algorithmNumber < 2; algorithmNumber++) {
        algorithm = algorithmNumber==0 ? pSha3Settings->algorithm1 : pSha3Settings->algorithm2;
        pElem = getXMLElement(pRoot,string("algorithm")+CommonFnc::toString(algorithmNumber+1));
        if (atoi(getXMLElementValue(pElem,"@type").c_str()) != algorithm) {
            mainLogger.out(LOGGER_ERROR) << "Incompatible algorithm types." << endl;
            return STAT_CONFIG_INCORRECT;
        }
        if (algorithm == SHA3_RANDOM) continue;
        if (atoi(getXMLElementValue(pElem,"hash_output_length").c_str()) != m_hashOutputLengths[algorithmNumber]) {
            mainLogger.out(LOGGER_ERROR) << "Incompatible hash output length." << endl;
            return STAT_CONFIG_INCORRECT;
        }
        istringstream counter(getXMLElementValue(pElem,"counter"));
        counter >> m_counters[algorithmNumber];
        m_usedBytes[algorithmNumber] = atoi(getXMLElementValue(pElem,"used_bytes").c_str());
        istringstream hash(getXMLElementValue(pElem,"current_hash"));
        for (int byte = 0; byte < m_hashOutputLengths[algorithmNumber]; byte++) {
            hash >> m_hashOutputs[algorithmNumber][byte];
            if (hash.fail()) status = STAT_CONFIG_INCORRECT;
        }
    }
    return status;
}

int Hasher::getTestVector(int algorithmNumber, unsigned char* tvInputs, unsigned char* tvOutputs) {
    int status = STAT_OK;
    if (algorithmNumber != 0 && algorithmNumber != 1) {
        mainLogger.out(LOGGER_ERROR) << "Incorrect algorithm number (" << algorithmNumber << ")." << endl;
        return STAT_INVALID_ARGUMETS;
    }
    int algorithm = algorithmNumber==0 ? pSha3Settings->algorithm1 : pSha3Settings->algorithm2;
    ofstream tvFile;
    if (pGlobals->settings->outputs.verbosity >= LOGGER_VERBOSITY_DEEP_DEBUG) {
        tvFile.open(FILE_TEST_VECTORS_HR, ios_base::app | ios_base::binary);
        if (!tvFile.is_open())
            mainLogger.out(LOGGER_WARNING) << "Cannot write to human-readable test vector file." << endl;
    }

    switch (pSha3Settings->plaintextType) {
    case SHA3_COUNTER:
        // set correct input
        if (algorithm != SHA3_RANDOM) {
            // create new hash output, if necessary
            if (m_hashOutputLengths[algorithmNumber] - m_usedBytes[algorithmNumber] < pGlobals->settings->testVectors.inputLength) {
                int hashStatus = SHA3_HASH_SUCCESS;
                hashStatus = m_hashFunctions[algorithmNumber]->Init(8*m_hashOutputLengths[algorithmNumber]);
                if (hashStatus != SHA3_HASH_SUCCESS) {
                    mainLogger.out(LOGGER_ERROR) << "Hash function " << Sha3Functions::sha3ToString(algorithm);
                    mainLogger.out() << " could not be initialized (status: " << hashStatus << ")." << endl;
                    return STAT_PROJECT_ERROR;
                }
                hashStatus = m_hashFunctions[algorithmNumber]->Update((const unsigned char*)&(m_counters[algorithmNumber]),
                                                                      8*sizeof(m_counters[algorithmNumber]));
                if (hashStatus != SHA3_HASH_SUCCESS) {
                    mainLogger.out(LOGGER_ERROR) << "Hash function " << Sha3Functions::sha3ToString(algorithm);
                    mainLogger.out() << " could not update hash (status: " << hashStatus << ")." << endl;
                    return STAT_PROJECT_ERROR;
                }
                hashStatus = m_hashFunctions[algorithmNumber]->Final(m_hashOutputs[algorithmNumber]);
                if (hashStatus != SHA3_HASH_SUCCESS) {
                    mainLogger.out(LOGGER_ERROR) << "Hash function " << Sha3Functions::sha3ToString(algorithm);
                    mainLogger.out() << " could not finalize hash (status: " << hashStatus << ")." << endl;
                    return STAT_PROJECT_ERROR;
                }
                m_usedBytes[algorithmNumber] = 0;
                tvFile << "NEW HASH CREATED" << endl << "Input: ";
                tvFile << setw(2*sizeof(m_counters[algorithmNumber])) << hex << m_counters[algorithmNumber] << endl;
                tvFile << "Output: ";
                for (int byte = 0; byte < m_hashOutputLengths[algorithmNumber]; byte++) {
                    tvFile << setfill('0') << setw(2) << hex << (int)(m_hashOutputs[algorithmNumber][byte]);
                }
                tvFile << endl;
                m_counters[algorithmNumber]++;
            }
            for (int byte = 0; byte < pGlobals->settings->testVectors.inputLength; byte++) {
                tvInputs[byte] = m_hashOutputs[algorithmNumber][m_usedBytes[algorithmNumber]+byte];
            }
            m_usedBytes[algorithmNumber] += pGlobals->settings->testVectors.inputLength;
        } else { // random data stream
            if (pGlobals->settings->outputs.verbosity >= LOGGER_VERBOSITY_DEEP_DEBUG)
                tvFile << "(RANDOM INPUT - " << rndGen->shortDescription() << "):" << endl;
            for (int input = 0; input < pGlobals->settings->testVectors.inputLength; input++) {
                rndGen->getRandomFromInterval(UCHAR_MAX, &(tvInputs[input]));
            }
        }

        // set correct output
        for (int output = 0; output < pGlobals->settings->testVectors.outputLength; output++)
            tvOutputs[output] = algorithmNumber * 0xff;
        break;
    default:
        mainLogger.out(LOGGER_ERROR) << "Unknown test vector generation method (";
        mainLogger.out() << pSha3Settings->plaintextType << ")." << endl;
        return STAT_INVALID_ARGUMETS;
    }

    // save test vector to human readable test vector file
    if (pGlobals->settings->outputs.verbosity >= LOGGER_VERBOSITY_DEEP_DEBUG) {
        tvFile.close();
    }
    return status;
}

void Hasher::checkNumRounds(int numRounds, int algorithmConstant) {
    if (numRounds > -1) {
        mainLogger.out(LOGGER_WARNING) << Sha3Functions::sha3ToString(algorithmConstant);
        mainLogger.out() << " cannot be limited in rounds - using UNLIMITTED VERSION!." << endl;
    }
}
