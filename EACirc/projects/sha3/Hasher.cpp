#include "Hasher.h"
#include "hash_functions/hashFunctions.h"

Hasher::Hasher() {
    int algorithm = -1;
    int numRounds = -1;

    for (int algorithmNumber=0; algorithmNumber<2; algorithmNumber++) {
        // get correct settings for this hash function
        switch (algorithmNumber) {
        case 0:
            algorithm = pSha3Settings->algorithm1;
            numRounds = pSha3Settings->limitAlgRounds ? pSha3Settings->alg1RoundsCount : -1;
            break;
        case 1:
            algorithm = pSha3Settings->algorithm1;
            numRounds = pSha3Settings->limitAlgRounds ? pSha3Settings->alg1RoundsCount : -1;
            break;
        default:
            mainLogger.out(LOGGER_ERROR) << "Unsupported Hasher iteration in initialization (";
            mainLogger.out() << algorithmNumber << ")." << endl;
        }
        // allocate hash functions
        m_hashFunctions[algorithmNumber] = NULL;
        m_hashOutputs[algorithmNumber] = NULL;
        m_counters[algorithmNumber] = 0;
        m_bitsUsed[algorithmNumber] = 0;
        if (algorithm == SHA3_RANDOM) continue;
        // if algorithm is set (other than random), allocate hash function
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
            //case SHA3_EDON:         m_hashFunctions[algorithmNumber] = new Edon(numRounds); break;        consturctor!
            //case SHA3_ENRUPT:       m_hashFunctions[algorithmNumber] = new EnRUPT(numRounds); break;
        case SHA3_ESSENCE:      m_hashFunctions[algorithmNumber] = new Essence(numRounds); break;
        case SHA3_FUGUE:        m_hashFunctions[algorithmNumber] = new Fugue(numRounds); break;
        case SHA3_GROSTL:       m_hashFunctions[algorithmNumber] = new Grostl(numRounds); break;
        case SHA3_HAMSI:        m_hashFunctions[algorithmNumber] = new Hamsi(numRounds); break;
        case SHA3_JH:           m_hashFunctions[algorithmNumber] = new JH(numRounds); break;
            //case SHA3_KECCAK:       m_hashFunctions[algorithmNumber] = new Keccak(numRounds); break;      consturctor!
            //case SHA3_KHICHIDI:     m_hashFunctions[algorithmNumber] = new Khichidi(numRounds); break;    consturctor!
        case SHA3_LANE:         m_hashFunctions[algorithmNumber] = new Lane(numRounds); break;
        case SHA3_LESAMNTA:     m_hashFunctions[algorithmNumber] = new Lesamnta(numRounds); break;
        case SHA3_LUFFA:        m_hashFunctions[algorithmNumber] = new Luffa(numRounds); break;
            //case SHA3_LUX:          m_hashFunctions[algorithmNumber] = new Lux(numRounds); break;
            //case SHA3_MSCSHA3:      m_hashFunctions[algorithmNumber] = new Mscsha(numRounds); break;      consturctor!
        case SHA3_MD6:          m_hashFunctions[algorithmNumber] = new MD6(numRounds); break;
        case SHA3_MESHHASH:     m_hashFunctions[algorithmNumber] = new MeshHash(numRounds); break;
            //case SHA3_NASHA:        m_hashFunctions[algorithmNumber] = new Nasha(numRounds); break;       consturctor!
            //case SHA3_SANDSTORM:    m_hashFunctions[algorithmNumber] = new Sandstorm(numRounds); break;
        case SHA3_SARMAL:       m_hashFunctions[algorithmNumber] = new Sarmal(numRounds); break;
            //case SHA3_SHABAL:       m_hashFunctions[algorithmNumber] = new Shabal(numRounds); break;      consturctor!
            //case SHA3_SHAMATA:      m_hashFunctions[algorithmNumber] = new Shamata(numRounds); break;     consturctor!
        case SHA3_SHAVITE3:     m_hashFunctions[algorithmNumber] = new SHAvite(numRounds); break;
        case SHA3_SIMD:         m_hashFunctions[algorithmNumber] = new Simd(numRounds); break;
            //case SHA3_SKEIN:        m_hashFunctions[algorithmNumber] = new Skein(numRounds); break;       consturctor!
            //case SHA3_SPECTRALHASH: m_hashFunctions[algorithmNumber] = new SpectralHash(numRounds); break;        consturctor!
            //case SHA3_STREAMHASH:   m_hashFunctions[algorithmNumber] = new StreamHash(numRounds); break;  consturctor!
            //case SHA3_SWIFFTX:      m_hashFunctions[algorithmNumber] = new Swifftx(numRounds); break;
        case SHA3_TANGLE:       m_hashFunctions[algorithmNumber] = new Tangle(numRounds); break;
            //case SHA3_TIB3:         m_hashFunctions[algorithmNumber] = new Tib3(numRounds); break;
        case SHA3_TWISTER:      m_hashFunctions[algorithmNumber] = new Twister(numRounds); break;
        case SHA3_VORTEX:       m_hashFunctions[algorithmNumber] = new Vortex(numRounds); break;
        case SHA3_WAMM:         m_hashFunctions[algorithmNumber] = new WaMM(numRounds); break;
        case SHA3_WATERFALL:    m_hashFunctions[algorithmNumber] = new Waterfall(numRounds); break;
        default:
            mainLogger.out(LOGGER_ERROR) << "Unknown hash function type (" << algorithm << ")." << endl;
            return;
        }
    }
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

int Hasher::getTestVector(int algorithmNumber, unsigned char* tvInputs, unsigned char* tvOutputs) {
    int status = STAT_OK;
    if (algorithmNumber != 1 && algorithmNumber != 2) {
        mainLogger.out(LOGGER_ERROR) << "Incorrect algorithm number (" << algorithmNumber << ")." << endl;
        return STAT_INVALID_ARGUMETS;
    }

    switch (pSha3Settings->vectorGenerationMethod) {
    case SHA3_COUNTER:
        return STAT_NOT_IMPLEMENTED_YET;
        // set correct inputput

        // set correct output

        break;
    default:
        mainLogger.out(LOGGER_ERROR) << "Unknown test vector generation method (";
        mainLogger.out() << pSha3Settings->vectorGenerationMethod << ")." << endl;
        return STAT_INVALID_ARGUMETS;
    }

    return status;
}
