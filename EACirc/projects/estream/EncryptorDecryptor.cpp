#include "EncryptorDecryptor.h"
#include "EstreamInterface.h"
#include "EstreamProject.h"
#include "garandom.h"
#include <string>

EncryptorDecryptor::EncryptorDecryptor() : m_setIV(false), m_setKey(false) {
    int algorithm = -1;
    int numRounds = -1;

    for (int cipherNumber=0; cipherNumber<2; cipherNumber++) {
        // get correct settings for this cipher
        switch (cipherNumber) {
        case 0:
            algorithm = pEstreamSettings->algorithm1;
            numRounds = pEstreamSettings->limitAlgRounds ? pEstreamSettings->alg1RoundsCount : -1;
            break;
        case 1:
            algorithm = pEstreamSettings->algorithm1;
            numRounds = pEstreamSettings->limitAlgRounds ? pEstreamSettings->alg1RoundsCount : -1;
            break;
        default:
            mainLogger.out(LOGGER_ERROR) << "Unsupported EncryptorDecryptor iteration in initialization (";
            mainLogger.out() << cipherNumber << ")." << endl;
        }
        // allocate ciphers and internalStates
        for (int streamNumber=0; streamNumber<2; streamNumber++) {
            m_ciphers[cipherNumber][streamNumber] = NULL;
            m_internalStates[cipherNumber][streamNumber] = NULL;
            if (algorithm == ESTREAM_RANDOM) continue;

            // if algorithm is set (other than random), allocate ciphers and internalState
            switch (algorithm) {
            case ESTREAM_ABC:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_ABC();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(ABC_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(ABC_ctx));
                break;
            case ESTREAM_ACHTERBAHN:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Achterbahn();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(ACHTERBAHN_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(ACHTERBAHN_ctx));
                break;
            case ESTREAM_CRYPTMT:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Cryptmt();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(CRYPTMT_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(CRYPTMT_ctx));
                break;
            case ESTREAM_DECIM:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Decim();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(DECIM_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(DECIM_ctx));
                if (numRounds == -1) numRounds = 8;
                break;
            case ESTREAM_DICING:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Dicing();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(DICING_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(DICING_ctx));
                break;
            case ESTREAM_DRAGON:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Dragon();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(DRAGON_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(DRAGON_ctx));
                break;
            case ESTREAM_EDON80:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Edon80();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(EDON80_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(EDON80_ctx));
                break;
            case ESTREAM_FFCSR:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_FFCSR();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(FFCSR_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(FFCSR_ctx));
                break;
            case ESTREAM_FUBUKI:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Fubuki();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(FUBUKI_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(FUBUKI_ctx));
                if (numRounds == -1) numRounds = 4;
                break;
            case ESTREAM_GRAIN:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Grain();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(GRAIN_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(GRAIN_ctx));
                if (numRounds == -1) numRounds = 13;
                break;
            case ESTREAM_HC128:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_HC128();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(HC128_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(HC128_ctx));
                break;
            case ESTREAM_HERMES:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Hermes();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(HERMES_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(HERMES_ctx));
                if (numRounds == -1) numRounds = 10;
                break;
            case ESTREAM_LEX:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Lex();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(LEX_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(LEX_ctx));
                // typically 10 rounds, up to 14 possible, but internal constant would have to be changed
                if (numRounds == -1) numRounds = 10;
                break;
            case ESTREAM_MAG:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Mag();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(MAG_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(MAG_ctx));
                break;
            case ESTREAM_MICKEY:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Mickey();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(MICKEY_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(MICKEY_ctx));
                break;
            case ESTREAM_MIR1:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Mir();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(MIR1_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(MIR1_ctx));
                break;
            case ESTREAM_POMARANCH:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Pomaranch();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(POMARANCH_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(POMARANCH_ctx));
                break;
            case ESTREAM_PY:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Py();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(PY_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(PY_ctx));
                break;
            case ESTREAM_RABBIT:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Rabbit();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(RABBIT_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(RABBIT_ctx));
                break;
            case ESTREAM_SALSA20:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Salsa();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(SALSA_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(SALSA_ctx));
                if (numRounds == -1) numRounds = 12;
                break;
            case ESTREAM_SFINKS:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Sfinks();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(SFINKS_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(SFINKS_ctx));
                break;
            case ESTREAM_SOSEMANUK:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Sosemanuk();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(SOSEMANUK_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(SOSEMANUK_ctx));
                break;
            case ESTREAM_TRIVIUM:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Trivium();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(TRIVIUM_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(TRIVIUM_ctx));
                break;
            case ESTREAM_TSC4:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Tsc4();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(TSC4_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(TSC4_ctx));
                if (numRounds == -1) numRounds = 32;
                break;
            case ESTREAM_WG:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Wg();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(WG_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(WG_ctx));
                break;
            case ESTREAM_YAMB:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Yamb();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(YAMB_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(YAMB_ctx));
                break;
            case ESTREAM_ZKCRYPT:
                m_ciphers[cipherNumber][streamNumber] = new ECRYPT_Zkcrypt();
                m_internalStates[cipherNumber][streamNumber] = (void*)malloc(sizeof(ZKCRYPT_ctx));
                memset(m_internalStates[cipherNumber][streamNumber],0,sizeof(ZKCRYPT_ctx));
                break;
            default:
                mainLogger.out(LOGGER_ERROR) << "Unknown cipher type (" << algorithm << ")." << endl;
                return;
            }
            if (numRounds == -1) {
                mainLogger.out(LOGGER_WARNING) << "Number of rounds probably incorrectly set (" << numRounds;
                mainLogger.out() << "). See code and/or manual." << endl;
            }
            m_ciphers[cipherNumber][streamNumber]->numRounds = numRounds;
            m_ciphers[cipherNumber][streamNumber]->ECRYPT_init();
        }
    }
}

EncryptorDecryptor::~EncryptorDecryptor() {
    for (int cipherNumber=0; cipherNumber<2; cipherNumber++) {
        for (int streamNumber=0; streamNumber<2; streamNumber++) {
            if (m_ciphers[cipherNumber][streamNumber]) {
                delete m_ciphers[cipherNumber][streamNumber];
                m_ciphers[cipherNumber][streamNumber] = NULL;
            }
            if (m_internalStates[cipherNumber][streamNumber]) {
                free(m_internalStates[cipherNumber][streamNumber]);
                m_internalStates[cipherNumber][streamNumber] = NULL;
            }
        }
    }
}

int EncryptorDecryptor::setupIV() {
    switch (pEstreamSettings->ivType) {
    case ESTREAM_GENTYPE_ZEROS:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) m_iv[input] = 0x00;
        break;
    case ESTREAM_GENTYPE_ONES:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) m_iv[input] = 0x01;
        break;
    case ESTREAM_GENTYPE_RANDOM:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) rndGen->getRandomFromInterval(255, &(m_iv[input]));
        break;
    case ESTREAM_GENTYPE_BIASRANDOM:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) biasRndGen->getRandomFromInterval(255, &(m_iv[input]));
        break;
    default:
        mainLogger.out(LOGGER_ERROR) << "Unknown IV type (" << pEstreamSettings->ivType << ")." << endl;
        return STAT_INVALID_ARGUMETS;
    }

    // human-readable test vector logging
    if (pGlobals->settings->testVectors.saveTestVectors) {
        ofstream tvFile(FILE_TEST_VECTORS_HR, ios::app);
        tvFile << setfill('0');
        tvFile << "setting IV: ";
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++)
            tvFile << setw(2) << hex << (int)(m_iv[input]);
        tvFile << endl;
        tvFile.close();
    }

    if (pEstreamSettings->algorithm1 != ESTREAM_RANDOM) {
        m_ciphers[0][0]->ECRYPT_ivsetup(m_internalStates[0][0], m_iv);
        m_ciphers[0][1]->ECRYPT_ivsetup(m_internalStates[0][1], m_iv);
    }
    if (pEstreamSettings->algorithm2 != ESTREAM_RANDOM) {
        m_ciphers[1][0]->ECRYPT_ivsetup(m_internalStates[1][0], m_iv);
        m_ciphers[1][1]->ECRYPT_ivsetup(m_internalStates[1][1], m_iv);
    }

    m_setIV = true;
    return STAT_OK;
}

int EncryptorDecryptor::setupKey() {
    switch (pEstreamSettings->keyType) {
    case ESTREAM_GENTYPE_ZEROS:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) m_key[input] = 0x00;
        break;
    case ESTREAM_GENTYPE_ONES:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) m_key[input] = 0x01;
        break;
    case ESTREAM_GENTYPE_RANDOM:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) rndGen->getRandomFromInterval(255, &(m_key[input]));
        break;
    case ESTREAM_GENTYPE_BIASRANDOM:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) biasRndGen->getRandomFromInterval(255, &(m_key[input]));
        break;
    default:
        mainLogger.out(LOGGER_ERROR) << "Unknown key type (" << pEstreamSettings->keyType << ")." << endl;
        return STAT_INVALID_ARGUMETS;
        break;
    }

    // human-readable test vector logging
    if (pGlobals->settings->testVectors.saveTestVectors) {
        ofstream tvFile(FILE_TEST_VECTORS_HR, ios::app);
        tvFile << setfill('0');
        tvFile << "setting key: ";
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++)
            tvFile << setw(2) << hex << (int)(m_key[input]);
        tvFile << endl;
        tvFile.close();
    }

    if (pEstreamSettings->algorithm1 != ESTREAM_RANDOM) {
        m_ciphers[0][0]->ECRYPT_keysetup(m_internalStates[0][0], m_key, STREAM_BLOCK_SIZE*8, STREAM_BLOCK_SIZE*8);
        m_ciphers[0][1]->ECRYPT_keysetup(m_internalStates[0][1], m_key, STREAM_BLOCK_SIZE*8, STREAM_BLOCK_SIZE*8);
    }
    if (pEstreamSettings->algorithm2 != ESTREAM_RANDOM) {
        m_ciphers[1][0]->ECRYPT_keysetup(m_internalStates[1][0], m_key, STREAM_BLOCK_SIZE*8, STREAM_BLOCK_SIZE*8);
        m_ciphers[1][1]->ECRYPT_keysetup(m_internalStates[1][1], m_key, STREAM_BLOCK_SIZE*8, STREAM_BLOCK_SIZE*8);
    }

    m_setKey = true;
    return STAT_OK;
}

int EncryptorDecryptor::encrypt(unsigned char* plain, unsigned char* cipher, int cipherNumber, int streamNumber, int length) {
    if (!m_setIV) {
        mainLogger.out(LOGGER_ERROR) << "Initialization vector is not set!" << endl;
        return STAT_PROJECT_ERROR;
    }
    if (!m_setKey) {
        mainLogger.out(LOGGER_ERROR) << "Key is not set!" << endl;
        return STAT_PROJECT_ERROR;
    }

    if (!length) length = pGlobals->settings->testVectors.inputLength;
    // WRONG IMPLEMENTATION OF ABC:
    //typeof hax
    if (dynamic_cast<ECRYPT_ABC*>(m_ciphers[cipherNumber][streamNumber]))
        ((ECRYPT_ABC*)m_ciphers[cipherNumber][streamNumber])->ABC_process_bytes(0,(ABC_ctx*)m_internalStates[cipherNumber][streamNumber],plain,cipher, length*8);
    else
        m_ciphers[cipherNumber][streamNumber]->ECRYPT_encrypt_bytes(m_internalStates[cipherNumber][streamNumber], plain, cipher, length);
    return STAT_OK;
}

int EncryptorDecryptor::decrypt(unsigned char* cipher, unsigned char* plain, int cipherNumber, int streamNumber, int length) {
    if (!m_setIV) {
        mainLogger.out(LOGGER_ERROR) << "Initialization vector is not set!" << endl;
        return STAT_PROJECT_ERROR;
    }
    if (!m_setKey) {
        mainLogger.out(LOGGER_ERROR) << "Key is not set!" << endl;
        return STAT_PROJECT_ERROR;
    }

    if (!length) length = pGlobals->settings->testVectors.inputLength;
    // WRONG IMPLEMENTATION OF ABC:
    //typeof hax
    if (dynamic_cast<ECRYPT_ABC*>(m_ciphers[cipherNumber][streamNumber]))
        ((ECRYPT_ABC*)m_ciphers[cipherNumber][streamNumber])->ABC_process_bytes(1,(ABC_ctx*)m_internalStates[cipherNumber][streamNumber],cipher,plain, length*8);
    else
        m_ciphers[cipherNumber][streamNumber]->ECRYPT_decrypt_bytes(m_internalStates[cipherNumber][streamNumber], cipher, plain, length);
    return STAT_OK;
}
