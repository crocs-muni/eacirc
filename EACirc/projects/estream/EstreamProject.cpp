#include "EstreamProject.h"
#include "EstreamConstants.h"

ESTREAM_SETTINGS* pEstreamSettings = NULL;

// TODO clean-up initialization section
EstreamProject::EstreamProject()
    : IProject(PROJECT_ESTREAM) {
    encryptorDecryptor = NULL;
    this->numstats = new int[2];
}

EstreamProject::~EstreamProject() {
    delete[] this->numstats;
    if (encryptorDecryptor) delete encryptorDecryptor;
    encryptorDecryptor = NULL;
}

string EstreamProject::shortDescription() const {
    return "eStream candidate ciphers";
}

int EstreamProject::loadProjectConfiguration(TiXmlNode* pRoot) {
    estreamSettings.testVectorEstreamMethod = atoi(getXMLElementValue(pRoot,"ESTREAM/ESTREAM_USAGE_TYPE").c_str());
    estreamSettings.testVectorEstream = atoi(getXMLElementValue(pRoot,"ESTREAM/ALGORITHM_1").c_str());
    estreamSettings.testVectorEstream2 = atoi(getXMLElementValue(pRoot,"ESTREAM/ALGORITHM_2").c_str());
    estreamSettings.testVectorBalance = atoi(getXMLElementValue(pRoot,"ESTREAM/BALLANCED_TEST_VECTORS").c_str());
    estreamSettings.limitAlgRounds = (atoi(getXMLElementValue(pRoot,"ESTREAM/LIMIT_NUM_OF_ROUNDS").c_str())) ? true : false;
    estreamSettings.limitAlgRoundsCount = atoi(getXMLElementValue(pRoot,"ESTREAM/ROUNDS_ALG_1").c_str());
    estreamSettings.limitAlgRoundsCount2 = atoi(getXMLElementValue(pRoot,"ESTREAM/ROUNDS_ALG_2").c_str());
    estreamSettings.estreamInputType= atoi(getXMLElementValue(pRoot,"ESTREAM/PLAINTEXT_TYPE").c_str());
    estreamSettings.estreamKeyType = atoi(getXMLElementValue(pRoot,"ESTREAM/KEY_TYPE").c_str());
    estreamSettings.estreamIVType = atoi(getXMLElementValue(pRoot,"ESTREAM/IV_TYPE").c_str());
    estreamSettings.cipherInitializationFrequency = atoi(getXMLElementValue(pRoot,"ESTREAM/CIPHER_INIT_FREQ").c_str());
    estreamSettings.generateStream = (atoi(getXMLElementValue(pRoot,"ESTREAM/GENERATE_STREAM").c_str())) ? true : false;
    pEstreamSettings = &estreamSettings;

    // bind project settings into global settings
    pGlobals->settings->project = (void*) pEstreamSettings;

    return STAT_OK;
}

int EstreamProject::initializeProject() {
    encryptorDecryptor = new EncryptorDecryptor;
    return STAT_OK;
}

int EstreamProject::initializeProjectState() {
    int status = STAT_OK;
    if (pEstreamSettings->cipherInitializationFrequency == ESTREAM_INIT_CIPHERS_ONCE) {
        status = encryptorDecryptor->setupKey();
        if (status != STAT_OK) return status;
        status = encryptorDecryptor->setupIV();
    }
    return status;
}

int EstreamProject::setupPlaintext() {
    switch (pEstreamSettings->estreamInputType) {
    case ESTREAM_GENTYPE_ZEROS:
        for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) plain[input] = 0x00;
        break;
    case ESTREAM_GENTYPE_ONES:
        for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) plain[input] = 0x01;
        break;
    case ESTREAM_GENTYPE_RANDOM:
        for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) rndGen->getRandomFromInterval(255, &(plain[input]));
        break;
    case ESTREAM_GENTYPE_BIASRANDOM:
        for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) biasRndGen->getRandomFromInterval(255, &(plain[input]));
        break;
    default:
        mainLogger.out() << "error: Unknown plaintext type for " << shortDescription() << endl;
        return STAT_INCOMPATIBLE_PARAMETER;
    }

    return STAT_OK;
}

int EstreamProject::generateTestVectors() {
    int status = STAT_OK;

    if (pEstreamSettings->generateStream) {
        status = generateCipherDataStream();
        mainLogger.out() << "warning: Generation of stream of cipher data finished." << endl;
    }

    if (pEstreamSettings->cipherInitializationFrequency == ESTREAM_INIT_CIPHERS_FOR_SET) {
        encryptorDecryptor->setupKey();
        encryptorDecryptor->setupIV();
    }

    // USED FOR BALANCING TEST VECTORS
    this->numstats[0] = 0;
    this->numstats[1] = 0;

    for (int testVectorNumber = 0; testVectorNumber < pGlobals->settings->testVectors.numTestVectors; testVectorNumber++) {
        if (pGlobals->settings->testVectors.saveTestVectors == 1) {
            ofstream tvfile(FILE_TEST_VECTORS_HR, ios::app);
            tvfile << "Test vector n." << dec << testVectorNumber << endl;
            tvfile.close();
        }

        if (pEstreamSettings->cipherInitializationFrequency == ESTREAM_INIT_CIPHERS_FOR_VECTOR) {
            encryptorDecryptor->setupKey();
            encryptorDecryptor->setupIV();
        }
        if (getTestVector() != STAT_OK) {
            return STAT_PROJECT_ERROR;
        }

        for (int input = 0; input < MAX_INPUTS; input++) {
            pGlobals->testVectors[testVectorNumber][input] = inputs[input];
        }
        for (int output = 0; output < pGlobals->settings->circuit.sizeOutputLayer; output++)
            pGlobals->testVectors[testVectorNumber][MAX_INPUTS+output] = outputs[output];
    }
    return status;
}

int EstreamProject::getTestVector(){
    int status = STAT_OK;
    ofstream tvFile(FILE_TEST_VECTORS_HR, ios::app);
    int streamnum = 0;
    switch (pEstreamSettings->testVectorEstreamMethod) {
        case ESTREAM_DISTINCT:

            //SHALL WE BALANCE TEST VECTORS?
            if (pEstreamSettings->testVectorBalance && (numstats[0] >= pGlobals->settings->testVectors.numTestVectors/2))
                streamnum = 1;
            else if (pEstreamSettings->testVectorBalance && (numstats[1] >= pGlobals->settings->testVectors.numTestVectors/2))
                streamnum = 0;
            else
                rndGen->getRandomFromInterval(1, &streamnum);
            numstats[streamnum]++;
            //Signalize the correct value
            for (int output = 0; output < pGlobals->settings->circuit.sizeOutputLayer; output++) outputs[output] = streamnum * 0xff;

            //generate the plaintext for stream
            if ((streamnum == 0 && pEstreamSettings->testVectorEstream != ESTREAM_RANDOM) ||
                (streamnum == 1 && pEstreamSettings->testVectorEstream2 != ESTREAM_RANDOM) ) {
                if (pGlobals->settings->testVectors.saveTestVectors == 1)
                    tvFile  << "(alg n." << ((streamnum==0)?pEstreamSettings->testVectorEstream:pEstreamSettings->testVectorEstream2) << " - " << ((streamnum==0)?pEstreamSettings->limitAlgRoundsCount:pEstreamSettings->limitAlgRoundsCount2) << " rounds): ";

                status = setupPlaintext();
                if (status != STAT_OK) return status;
                status = encryptorDecryptor->encrypt(plain,inputs,streamnum);
                if (status != STAT_OK) return status;
                status = encryptorDecryptor->decrypt(inputs,outplain,streamnum+2);
                if (status != STAT_OK) return status;

                // check if plaintext = encrypted-decrypted plaintext
                for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) {
                    if (outplain[input] != plain[input]) {
                        mainLogger.out() << "error: Decrypted text doesn't match the input. See " << FILE_TEST_VECTORS_HR << " for details." << endl;
                        tvFile << "### ERROR: PLAINTEXT-ENCDECTEXT MISMATCH!" << endl;
                        break;
                    }
                }
            }
            else { // RANDOM
                if (pGlobals->settings->testVectors.saveTestVectors == 1)
                    tvFile << "(RANDOM INPUT - " << rndGen->shortDescription() << "):";
                for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) {
                    rndGen->getRandomFromInterval(255, &outplain[input]);
                    plain[input] = inputs[input] = outplain[input];
                }
            }
            break;

    case ESTREAM_BITS_TO_CHANGE:
        status = setupPlaintext();
        if (status != STAT_OK) return status;

        // WE NEED TO LET EVALUATOR KNOW THE INPUTS
        for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++)
            outputs[input] = inputs[input];

        break;

    default:
        mainLogger.out() << "error: unknown testVectorEstreamMethod (" << pEstreamSettings->testVectorEstreamMethod << ") in " << shortDescription() << endl;
        return STAT_INCOMPATIBLE_PARAMETER;
        break;
    }

    // SAVE TEST VECTORS IN BINARY FILES
    if (pGlobals->settings->testVectors.saveTestVectors) {
        if (streamnum == 0) {
            ofstream itvfile(FILE_TEST_DATA_1, ios::app | ios::binary);
            for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) {
                    itvfile << inputs[input];
            }
            itvfile.close();
        }
        else {
            ofstream itvfile(FILE_TEST_DATA_2, ios::app | ios::binary);
            for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) {
                    itvfile << inputs[input];
            }
            itvfile.close();
        }
    }

    // save human-readable test vector
    if (pGlobals->settings->testVectors.saveTestVectors) {
        int tvg = 0;
        if (streamnum == 0) tvg = pEstreamSettings->testVectorEstream;
        else tvg = pEstreamSettings->testVectorEstream2;
        tvFile << setfill('0');

        if (memcmp(inputs,plain,pGlobals->settings->testVectors.testVectorLength) != 0) {
            for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++)
            tvFile << setw(2) << hex << (int)(plain[input]);
            tvFile << "::";
        }

        for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++)
            tvFile << setw(2) << hex << (int)(inputs[input]);

        if (memcmp(inputs,outplain,pGlobals->settings->testVectors.testVectorLength) != 0) {
            tvFile << "::";
            for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++)
            tvFile << setw(2) << hex << (int)(outplain[input]);
        }
        tvFile << endl;
    }
    tvFile.close();

    return STAT_OK;
}

int EstreamProject::generateCipherDataStream() {
    if (pEstreamSettings->testVectorEstream == ESTREAM_RANDOM) {
        mainLogger.out() << "error: You cannot generate stream from random, you must specify a cipher." << endl;
        return STAT_INCOMPATIBLE_PARAMETER;
    } else {
        mainLogger.out() << "info: You are generating stream for " << estreamToString(pEstreamSettings->testVectorEstream);
        if (!pEstreamSettings->limitAlgRounds) {
            mainLogger.out() << " (unlimitted version)" << endl;
        } else {
            mainLogger.out() << " (" << pEstreamSettings->limitAlgRoundsCount << " rounds)" << endl;
        }
    }

    int status = STAT_OK;
    int streamnum = 0;

    if (pEstreamSettings->cipherInitializationFrequency == ESTREAM_INIT_CIPHERS_ONCE) {
        status = encryptorDecryptor->setupKey();
        if (status != STAT_OK) return status;
        encryptorDecryptor->setupIV();
        if (status != STAT_OK) return status;
    }

    while (true) {
        if (pEstreamSettings->cipherInitializationFrequency == ESTREAM_INIT_CIPHERS_FOR_SET) {
            status = encryptorDecryptor->setupKey();
            if (status != STAT_OK) return status;
            encryptorDecryptor->setupIV();
            if (status != STAT_OK) return status;
        }

        for (int testVectorNumber = 0; testVectorNumber < pGlobals->settings->testVectors.numTestVectors; testVectorNumber++) {
            if (pEstreamSettings->cipherInitializationFrequency == ESTREAM_INIT_CIPHERS_FOR_VECTOR) {
                status = encryptorDecryptor->setupKey();
                if (status != STAT_OK) return status;
                encryptorDecryptor->setupIV();
                if (status != STAT_OK) return status;
            }

            status = setupPlaintext();
            if (status != STAT_OK) return status;
            status = encryptorDecryptor->encrypt(plain,inputs,streamnum);
            if (status != STAT_OK) return status;
            status = encryptorDecryptor->decrypt(inputs,outplain,streamnum+2);
            if (status != STAT_OK) return status;

            // check if plaintext = encrypted-decrypted plaintext
            for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) {
                if (outplain[input] != plain[input]) {
                    mainLogger.out() << "error: Decrypted text doesn't match the input." << endl;
                    break;
                }
            }

            for (int index = 0; index < pGlobals->settings->testVectors.testVectorLength; index++) {
                cout << inputs[index];
            }
        }
    }
}
