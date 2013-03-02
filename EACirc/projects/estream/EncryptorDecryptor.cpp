#include "EncryptorDecryptor.h"
#include "EstreamInterface.h"
#include "EstreamProject.h"
#include "garandom.h"
#include <string>

EncryptorDecryptor::EncryptorDecryptor() : m_setIV(false), m_setKey(false) {
    int numRounds = (pEstreamSettings->limitAlgRounds) ? pEstreamSettings->alg1RoundsCount : -1;
    int numRounds2 = (pEstreamSettings->limitAlgRounds) ? pEstreamSettings->alg2RoundsCount : -1;
    for(int i = 0; i < 4; i++) {
        ctxarr[i] = NULL;
        ecryptarr[i] = NULL;
    }
	
    int testVectorAlg = pEstreamSettings->algorithm1;
	int nR = numRounds;

	for (int i=0; i<2; i++) {
	   if (i == 1) {
               testVectorAlg = pEstreamSettings->algorithm2;
			   nR = numRounds2;
		}
       switch (testVectorAlg) {
		   case ESTREAM_ABC: {
				ECRYPT_ABC* ecryptx = new ECRYPT_ABC();
				ABC_ctx* ctxa = (ABC_ctx*)malloc(sizeof(ABC_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_ABC* ecryptx2 = new ECRYPT_ABC();
				ABC_ctx* ctxa2 = (ABC_ctx*)malloc(sizeof(ABC_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;
		   }
		   case ESTREAM_ACHTERBAHN: {
				ECRYPT_Achterbahn* ecryptx = new ECRYPT_Achterbahn();
				ACHTERBAHN_ctx* ctxa = (ACHTERBAHN_ctx*)malloc(sizeof(ACHTERBAHN_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Achterbahn* ecryptx2 = new ECRYPT_Achterbahn();
				ACHTERBAHN_ctx* ctxa2 = (ACHTERBAHN_ctx*)malloc(sizeof(ACHTERBAHN_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_CRYPTMT: {
				ECRYPT_Cryptmt* ecryptx = new ECRYPT_Cryptmt();
				CRYPTMT_ctx* ctxa = (CRYPTMT_ctx*)malloc(sizeof(CRYPTMT_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Cryptmt* ecryptx2 = new ECRYPT_Cryptmt();
				CRYPTMT_ctx* ctxa2 = (CRYPTMT_ctx*)malloc(sizeof(CRYPTMT_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_DECIM: {
				ECRYPT_Decim* ecryptx = new ECRYPT_Decim();
				DECIM_ctx* ctxa = (DECIM_ctx*)malloc(sizeof(DECIM_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Decim* ecryptx2 = new ECRYPT_Decim();
				DECIM_ctx* ctxa2 = (DECIM_ctx*)malloc(sizeof(DECIM_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
           // if unlimited, set correct number of round
                if (nR == -1) nR = 8;
				break;}
		   case ESTREAM_DICING: {
				ECRYPT_Dicing* ecryptx = new ECRYPT_Dicing();
				DICING_ctx* ctxa = (DICING_ctx*)malloc(sizeof(DICING_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Dicing* ecryptx2 = new ECRYPT_Dicing();
				DICING_ctx* ctxa2 = (DICING_ctx*)malloc(sizeof(DICING_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_DRAGON: {
				ECRYPT_Dragon* ecryptx = new ECRYPT_Dragon();
				DRAGON_ctx* ctxa = (DRAGON_ctx*)malloc(sizeof(DRAGON_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Dragon* ecryptx2 = new ECRYPT_Dragon();
				DRAGON_ctx* ctxa2 = (DRAGON_ctx*)malloc(sizeof(DRAGON_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_EDON80: {
				ECRYPT_Edon80* ecryptx = new ECRYPT_Edon80();
				EDON80_ctx* ctxa = (EDON80_ctx*)malloc(sizeof(EDON80_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Edon80* ecryptx2 = new ECRYPT_Edon80();
				EDON80_ctx* ctxa2 = (EDON80_ctx*)malloc(sizeof(EDON80_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_FFCSR: {
				ECRYPT_FFCSR* ecryptx = new ECRYPT_FFCSR();
				FFCSR_ctx* ctxa = (FFCSR_ctx*)malloc(sizeof(FFCSR_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_FFCSR* ecryptx2 = new ECRYPT_FFCSR();
				FFCSR_ctx* ctxa2 = (FFCSR_ctx*)malloc(sizeof(FFCSR_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_FUBUKI: {
				ECRYPT_Fubuki* ecryptx = new ECRYPT_Fubuki();
				FUBUKI_ctx* ctxa = (FUBUKI_ctx*)malloc(sizeof(FUBUKI_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Fubuki* ecryptx2 = new ECRYPT_Fubuki();
				FUBUKI_ctx* ctxa2 = (FUBUKI_ctx*)malloc(sizeof(FUBUKI_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
		   // if unlimited, set correct number of round
                if (nR == -1) nR = 4;
				break;}
		   case ESTREAM_GRAIN: {
				ECRYPT_Grain* ecryptx = new ECRYPT_Grain();
				GRAIN_ctx* ctxa = (GRAIN_ctx*)malloc(sizeof(GRAIN_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Grain* ecryptx2 = new ECRYPT_Grain();
				GRAIN_ctx* ctxa2 = (GRAIN_ctx*)malloc(sizeof(GRAIN_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
           // if unlimited, set correct number of round
                if (nR == -1) nR = 13;
				break;}
		   case ESTREAM_HC128: {
				ECRYPT_HC128* ecryptx = new ECRYPT_HC128();
				HC128_ctx* ctxa = (HC128_ctx*)malloc(sizeof(HC128_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_HC128* ecryptx2 = new ECRYPT_HC128();
				HC128_ctx* ctxa2 = (HC128_ctx*)malloc(sizeof(HC128_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_HERMES: {
				ECRYPT_Hermes* ecryptx = new ECRYPT_Hermes();
				HERMES_ctx* ctxa = (HERMES_ctx*)malloc(sizeof(HERMES_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Hermes* ecryptx2 = new ECRYPT_Hermes();
				HERMES_ctx* ctxa2 = (HERMES_ctx*)malloc(sizeof(HERMES_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
		   // if unlimited, set correct number of round
                if (nR == -1) nR = 10;
				break;}
		   case ESTREAM_LEX: {
				ECRYPT_Lex* ecryptx = new ECRYPT_Lex();
				LEX_ctx* ctxa = (LEX_ctx*)malloc(sizeof(LEX_ctx));
           // clear LEX structures (for unsupported number of rounds)
                for (int i = 0; i<NUMWORDS ;i++) {
                    ctxa->ks[i] = 0;
                }
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Lex* ecryptx2 = new ECRYPT_Lex();
				LEX_ctx* ctxa2 = (LEX_ctx*)malloc(sizeof(LEX_ctx));
           // clear LEX structures (for unsupported number of rounds)
                for (int i = 0; i<NUMWORDS ;i++) {
                    ctxa2->ks[i] = 0;
                }
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
           // if unlimited, set correct number of round
                // typically 10 rounds, up to 14 possible, but internal constant would have to be changed
                if (nR == -1) nR = 10;
				break;}
		   case ESTREAM_MAG: {
				ECRYPT_Mag* ecryptx = new ECRYPT_Mag();
				MAG_ctx* ctxa = (MAG_ctx*)malloc(sizeof(MAG_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Mag* ecryptx2 = new ECRYPT_Mag();
				MAG_ctx* ctxa2 = (MAG_ctx*)malloc(sizeof(MAG_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_MICKEY: {
				ECRYPT_Mickey* ecryptx = new ECRYPT_Mickey();
				MICKEY_ctx* ctxa = (MICKEY_ctx*)malloc(sizeof(MICKEY_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Mickey* ecryptx2 = new ECRYPT_Mickey();
				MICKEY_ctx* ctxa2 = (MICKEY_ctx*)malloc(sizeof(MICKEY_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_MIR1: {
				ECRYPT_Mir* ecryptx = new ECRYPT_Mir();
				MIR1_ctx* ctxa = (MIR1_ctx*)malloc(sizeof(MIR1_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Mir* ecryptx2 = new ECRYPT_Mir();
				MIR1_ctx* ctxa2 = (MIR1_ctx*)malloc(sizeof(MIR1_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_POMARANCH: {
				ECRYPT_Pomaranch* ecryptx = new ECRYPT_Pomaranch();
				POMARANCH_ctx* ctxa = (POMARANCH_ctx*)malloc(sizeof(POMARANCH_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Pomaranch* ecryptx2 = new ECRYPT_Pomaranch();
				POMARANCH_ctx* ctxa2 = (POMARANCH_ctx*)malloc(sizeof(POMARANCH_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_PY: {
				ECRYPT_Py* ecryptx = new ECRYPT_Py();
				PY_ctx* ctxa = (PY_ctx*)malloc(sizeof(PY_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Py* ecryptx2 = new ECRYPT_Py();
				PY_ctx* ctxa2 = (PY_ctx*)malloc(sizeof(PY_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_RABBIT: {
				ECRYPT_Rabbit* ecryptx = new ECRYPT_Rabbit();
				RABBIT_ctx* ctxa = (RABBIT_ctx*)malloc(sizeof(RABBIT_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Rabbit* ecryptx2 = new ECRYPT_Rabbit();
				RABBIT_ctx* ctxa2 = (RABBIT_ctx*)malloc(sizeof(RABBIT_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_SALSA20: {
				ECRYPT_Salsa* ecryptx = new ECRYPT_Salsa();
				SALSA_ctx* ctxa = (SALSA_ctx*)malloc(sizeof(SALSA_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Salsa* ecryptx2 = new ECRYPT_Salsa();
				SALSA_ctx* ctxa2 = (SALSA_ctx*)malloc(sizeof(SALSA_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
           // if unlimited, set correct number of round
                if (nR == -1) nR = 12;
				break;}
		   case ESTREAM_SFINKS: {
				ECRYPT_Sfinks* ecryptx = new ECRYPT_Sfinks();
				SFINKS_ctx* ctxa = (SFINKS_ctx*)malloc(sizeof(SFINKS_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Sfinks* ecryptx2 = new ECRYPT_Sfinks();
				SFINKS_ctx* ctxa2 = (SFINKS_ctx*)malloc(sizeof(SFINKS_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_SOSEMANUK: {
				ECRYPT_Sosemanuk* ecryptx = new ECRYPT_Sosemanuk();
				SOSEMANUK_ctx* ctxa = (SOSEMANUK_ctx*)malloc(sizeof(SOSEMANUK_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Sosemanuk* ecryptx2 = new ECRYPT_Sosemanuk();
				SOSEMANUK_ctx* ctxa2 = (SOSEMANUK_ctx*)malloc(sizeof(SOSEMANUK_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_TRIVIUM: {
				ECRYPT_Trivium* ecryptx = new ECRYPT_Trivium();
				TRIVIUM_ctx* ctxa = (TRIVIUM_ctx*)malloc(sizeof(TRIVIUM_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Trivium* ecryptx2 = new ECRYPT_Trivium();
				TRIVIUM_ctx* ctxa2 = (TRIVIUM_ctx*)malloc(sizeof(TRIVIUM_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_TSC4: {
				ECRYPT_Tsc4* ecryptx = new ECRYPT_Tsc4();
				TSC4_ctx* ctxa = (TSC4_ctx*)malloc(sizeof(TSC4_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Tsc4* ecryptx2 = new ECRYPT_Tsc4();
				TSC4_ctx* ctxa2 = (TSC4_ctx*)malloc(sizeof(TSC4_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
           // if unlimited, set correct number of round
                if (nR == -1) nR = 32;
				break;}
		   case ESTREAM_WG: {
				ECRYPT_Wg* ecryptx = new ECRYPT_Wg();
				WG_ctx* ctxa = (WG_ctx*)malloc(sizeof(WG_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Wg* ecryptx2 = new ECRYPT_Wg();
				WG_ctx* ctxa2 = (WG_ctx*)malloc(sizeof(WG_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_YAMB: {
				ECRYPT_Yamb* ecryptx = new ECRYPT_Yamb();
				YAMB_ctx* ctxa = (YAMB_ctx*)malloc(sizeof(YAMB_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Yamb* ecryptx2 = new ECRYPT_Yamb();
				YAMB_ctx* ctxa2 = (YAMB_ctx*)malloc(sizeof(YAMB_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
				break;}
		   case ESTREAM_ZKCRYPT: {
				ECRYPT_Zkcrypt* ecryptx = new ECRYPT_Zkcrypt();
				ZKCRYPT_ctx* ctxa = (ZKCRYPT_ctx*)malloc(sizeof(ZKCRYPT_ctx));
				ecryptarr[i] = ecryptx;
				ctxarr[i] = (void*)ctxa;
				ECRYPT_Zkcrypt* ecryptx2 = new ECRYPT_Zkcrypt();
				ZKCRYPT_ctx* ctxa2 = (ZKCRYPT_ctx*)malloc(sizeof(ZKCRYPT_ctx));
				ecryptarr[2+i] = ecryptx2;
				ctxarr[2+i] = (void*)ctxa2;
			   break;}
		   case ESTREAM_RANDOM:
				ecryptarr[i] = NULL;
           // set correct number of rounds to prevent warning
                nR = 0;
				break;
		   default:
			   assert(false);
			   break;
		}

        if (nR == -1) {
            mainLogger.out() << "warning: Number of rounds probably incorrectly set (" << nR << "). See code and/or manual." << endl;
        }

        if (testVectorAlg != ESTREAM_RANDOM) {
			ecryptarr[i]->numRounds = nR;
			ecryptarr[i]->ECRYPT_init();
			ecryptarr[2+i]->numRounds = nR;
			ecryptarr[2+i]->ECRYPT_init();
		}

	}

    // generate header to human-readable test-vector file
    if (pGlobals->settings->testVectors.saveTestVectors) {
        ofstream tvFile(FILE_TEST_VECTORS_HR, ios::app);
        tvFile << "Using eStream ciphers and random generator to generate test vectors." << endl;
        tvFile << "  stream1: using " << estreamToString(pEstreamSettings->algorithm1);
        if (pEstreamSettings->limitAlgRounds) {
            tvFile << " (" << pEstreamSettings->alg1RoundsCount << " rounds)" << endl;
        } else {
            tvFile << " (unlimited version)" << endl;
        }
        tvFile << "  stream2: using " << estreamToString(pEstreamSettings->algorithm2);
        if (pEstreamSettings->limitAlgRounds) {
            tvFile << " (" << pEstreamSettings->alg2RoundsCount << " rounds)" << endl;
        } else {
            tvFile << " (unlimited version)" << endl;
        }
        tvFile << "Test vectors formatted as PLAINTEXT::CIPHERTEXT::DECRYPTED" << endl;
        tvFile.close();
    }
}

EncryptorDecryptor::~EncryptorDecryptor() {
    for(int i = 0; i < 4; i++) {
        if (ecryptarr[i] != NULL) {
            delete ecryptarr[i];
            ecryptarr[i] = NULL;
        }
        if (ctxarr[i] != NULL) {
            free(ctxarr[i]);
            ctxarr[i] = NULL;
        }
    }
}

int EncryptorDecryptor::setupIV() {
    switch (pEstreamSettings->ivType) {
    case ESTREAM_GENTYPE_ZEROS:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) iv[input] = 0x00;
        break;
    case ESTREAM_GENTYPE_ONES:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) iv[input] = 0x01;
        break;
    case ESTREAM_GENTYPE_RANDOM:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) rndGen->getRandomFromInterval(255, &(iv[input]));
        break;
    case ESTREAM_GENTYPE_BIASRANDOM:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) biasRndGen->getRandomFromInterval(255, &(iv[input]));
        break;
    default:
        mainLogger.out() << "error: Unknown IV type (" << pEstreamSettings->ivType << ")." << endl;
        return STAT_INCOMPATIBLE_PARAMETER;
    }

    // human-readable test vector logging
    if (pGlobals->settings->testVectors.saveTestVectors) {
        ofstream tvFile(FILE_TEST_VECTORS_HR, ios::app);
        tvFile << setfill('0');
        tvFile << "setting IV: ";
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++)
            tvFile << setw(2) << hex << (int)(iv[input]);
        tvFile << endl;
        tvFile.close();
    }

    if (pEstreamSettings->algorithm1 != ESTREAM_RANDOM) {
        ecryptarr[0]->ECRYPT_ivsetup(ctxarr[0], iv);
        ecryptarr[2]->ECRYPT_ivsetup(ctxarr[2], iv);
    }
    if (pEstreamSettings->algorithm2 != ESTREAM_RANDOM) {
        ecryptarr[1]->ECRYPT_ivsetup(ctxarr[1], iv);
        ecryptarr[3]->ECRYPT_ivsetup(ctxarr[3], iv);
    }

    m_setIV = true;
    return STAT_OK;
}

int EncryptorDecryptor::setupKey() {
    switch (pEstreamSettings->keyType) {
    case ESTREAM_GENTYPE_ZEROS:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) key[input] = 0x00;
        break;
    case ESTREAM_GENTYPE_ONES:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) key[input] = 0x01;
        break;
    case ESTREAM_GENTYPE_RANDOM:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) rndGen->getRandomFromInterval(255, &(key[input]));
        break;
    case ESTREAM_GENTYPE_BIASRANDOM:
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++) biasRndGen->getRandomFromInterval(255, &(key[input]));
        break;
    default:
        mainLogger.out() << "error: Unknown key type (" << pEstreamSettings->keyType << ")." << endl;
        return STAT_INCOMPATIBLE_PARAMETER;
        break;
    }

    // human-readable test vector logging
    if (pGlobals->settings->testVectors.saveTestVectors) {
        ofstream tvFile(FILE_TEST_VECTORS_HR, ios::app);
        tvFile << setfill('0');
        tvFile << "setting key: ";
        for (int input = 0; input < STREAM_BLOCK_SIZE; input++)
            tvFile << setw(2) << hex << (int)(key[input]);
        tvFile << endl;
        tvFile.close();
    }

    if (pEstreamSettings->algorithm1 != ESTREAM_RANDOM) {
        ecryptarr[0]->ECRYPT_keysetup(ctxarr[0], key, STREAM_BLOCK_SIZE*8, STREAM_BLOCK_SIZE*8);
        ecryptarr[2]->ECRYPT_keysetup(ctxarr[2], key, STREAM_BLOCK_SIZE*8, STREAM_BLOCK_SIZE*8);
    }
    if (pEstreamSettings->algorithm2 != ESTREAM_RANDOM) {
        ecryptarr[1]->ECRYPT_keysetup(ctxarr[1], key, STREAM_BLOCK_SIZE*8, STREAM_BLOCK_SIZE*8);
        ecryptarr[3]->ECRYPT_keysetup(ctxarr[3], key, STREAM_BLOCK_SIZE*8, STREAM_BLOCK_SIZE*8);
    }

    m_setKey = true;
    return STAT_OK;
}

int EncryptorDecryptor::encrypt(unsigned char* plain, unsigned char* cipher, int streamnum, int length) {
    if (!m_setIV) {
        mainLogger.out() << "error: Initialization vector is not set!" << endl;
        return STAT_PROJECT_ERROR;
    }
    if (!m_setKey) {
        mainLogger.out() << "error: Key is not set!" << endl;
        return STAT_PROJECT_ERROR;
    }

    if (!length) length = pGlobals->settings->testVectors.testVectorLength;
	// WRONG IMPLEMENTATION OF ABC:
	//typeof hax
	if (dynamic_cast<ECRYPT_ABC*>(ecryptarr[streamnum]))
		((ECRYPT_ABC*)ecryptarr[streamnum])->ABC_process_bytes(0,(ABC_ctx*)ctxarr[streamnum],plain,cipher, length*8);
	else
		ecryptarr[streamnum]->ECRYPT_encrypt_bytes(ctxarr[streamnum], plain, cipher, length);
    return STAT_OK;
}

int EncryptorDecryptor::decrypt(unsigned char* cipher, unsigned char* plain, int streamnum, int length) {
    if (!m_setIV) {
        mainLogger.out() << "error: Initialization vector is not set!" << endl;
        return STAT_PROJECT_ERROR;
    }
    if (!m_setKey) {
        mainLogger.out() << "error: Key is not set!" << endl;
        return STAT_PROJECT_ERROR;
    }

    if (!length) length = pGlobals->settings->testVectors.testVectorLength;
	// WRONG IMPLEMENTATION OF ABC:
    //typeof hax
    if (dynamic_cast<ECRYPT_ABC*>(ecryptarr[streamnum]))
		((ECRYPT_ABC*)ecryptarr[streamnum])->ABC_process_bytes(1,(ABC_ctx*)ctxarr[streamnum],cipher,plain, length*8);
	else
		ecryptarr[streamnum]->ECRYPT_decrypt_bytes(ctxarr[streamnum], cipher, plain, length);
    return STAT_OK;
}
