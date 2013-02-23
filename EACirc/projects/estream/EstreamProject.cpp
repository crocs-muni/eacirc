#include "EstreamProject.h"
#include "EstreamConstants.h"

ESTREAM_SETTINGS* pEstreamSettings = NULL;

// TODO clean-up initialization section
EstreamProject::EstreamProject()
    : IProject(PROJECT_ESTREAM) {
    memset(inputs,0,MAX_INPUTS);
    memset(outputs,0,MAX_OUTPUTS);
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
    estreamSettings.estreamUsageType = atoi(getXMLElementValue(pRoot,"ESTREAM/ESTREAM_USAGE_TYPE").c_str());
    estreamSettings.algorithm1 = atoi(getXMLElementValue(pRoot,"ESTREAM/ALGORITHM_1").c_str());
    estreamSettings.algorithm2 = atoi(getXMLElementValue(pRoot,"ESTREAM/ALGORITHM_2").c_str());
    estreamSettings.ballancedTestVectors = (atoi(getXMLElementValue(pRoot,"ESTREAM/BALLANCED_TEST_VECTORS").c_str())) ? true : false;
    estreamSettings.limitAlgRounds = (atoi(getXMLElementValue(pRoot,"ESTREAM/LIMIT_NUM_OF_ROUNDS").c_str())) ? true : false;
    estreamSettings.alg1RoundsCount = atoi(getXMLElementValue(pRoot,"ESTREAM/ROUNDS_ALG_1").c_str());
    estreamSettings.alg2RoundsCount = atoi(getXMLElementValue(pRoot,"ESTREAM/ROUNDS_ALG_2").c_str());
    estreamSettings.plaintextType= atoi(getXMLElementValue(pRoot,"ESTREAM/PLAINTEXT_TYPE").c_str());
    estreamSettings.keyType = atoi(getXMLElementValue(pRoot,"ESTREAM/KEY_TYPE").c_str());
    estreamSettings.ivType = atoi(getXMLElementValue(pRoot,"ESTREAM/IV_TYPE").c_str());
    estreamSettings.cipherInitializationFrequency = atoi(getXMLElementValue(pRoot,"ESTREAM/CIPHER_INIT_FREQ").c_str());
    estreamSettings.generateStream = (atoi(getXMLElementValue(pRoot,"ESTREAM/GENERATE_STREAM").c_str())) ? true : false;
    istringstream ss(getXMLElementValue(pRoot,"ESTREAM/STREAM_SIZE"));
    ss >> estreamSettings.streamSize;
    pEstreamSettings = &estreamSettings;

    // bind project settings into global settings
    pGlobals->settings->project = (void*) pEstreamSettings;

    return STAT_OK;
}

int EstreamProject::initializeProject() {
    encryptorDecryptor = new EncryptorDecryptor;
    // write project config to test vector file
    ofstream tvFile;
    tvFile.open(FILE_TEST_VECTORS, ios_base::app);
    if (!tvFile.is_open()) {
        mainLogger.out() << "error: Cannot write file for test vectors (" << FILE_TEST_VECTORS << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    tvFile << pGlobals->settings->main.projectType << " \t\t(project: " << shortDescription() << ")" << endl;
    tvFile << pEstreamSettings->estreamUsageType << " \t\t(eStream usage type)" << endl;
    tvFile << pEstreamSettings->cipherInitializationFrequency << " \t\t(cipher initialization frequency)" << endl;
    tvFile << pEstreamSettings->algorithm1 << " \t\t(algorithm1: " << estreamToString(pEstreamSettings->algorithm1) << ")" << endl;
    tvFile << pEstreamSettings->algorithm2 << " \t\t(algorithm2: " << estreamToString(pEstreamSettings->algorithm2) << ")" << endl;
    tvFile << pEstreamSettings->ballancedTestVectors << " \t\t(ballanced test vectors?)" << endl;
    tvFile << pEstreamSettings->limitAlgRounds << " \t\t(limit algorithm rounds?)" << endl;
    if (pEstreamSettings->limitAlgRounds) {
        tvFile << pEstreamSettings->alg1RoundsCount << " \t\t(algorithm1: " << pEstreamSettings->alg1RoundsCount << " rounds)" << endl;
        tvFile << pEstreamSettings->alg2RoundsCount << " \t\t(algorithm2: " << pEstreamSettings->alg2RoundsCount << " rounds)" << endl;
    }
    tvFile << pEstreamSettings->plaintextType << " \t\t(plaintext type)" << endl;
    tvFile << pEstreamSettings->keyType << " \t\t(key type)" << endl;
    tvFile << pEstreamSettings->ivType << " \t\t(IV type)" << endl;
    tvFile.close();
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
    switch (pEstreamSettings->plaintextType) {
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

    // if set so, do not generate test vectors but generate data strean to cout
    if (pEstreamSettings->generateStream) {
        status = generateCipherDataStream();
        if (status != STAT_OK) {
            return status;
        } else {
            return STAT_INTENTIONAL_EXIT;
        }
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
    switch (pEstreamSettings->estreamUsageType) {
        case ESTREAM_DISTINCT:

            //SHALL WE BALANCE TEST VECTORS?
            if (pEstreamSettings->ballancedTestVectors && (numstats[0] >= pGlobals->settings->testVectors.numTestVectors/2))
                streamnum = 1;
            else if (pEstreamSettings->ballancedTestVectors && (numstats[1] >= pGlobals->settings->testVectors.numTestVectors/2))
                streamnum = 0;
            else
                rndGen->getRandomFromInterval(1, &streamnum);
            numstats[streamnum]++;
            //Signalize the correct value
            for (int output = 0; output < pGlobals->settings->circuit.sizeOutputLayer; output++) outputs[output] = streamnum * 0xff;

            //generate the plaintext for stream
            if ((streamnum == 0 && pEstreamSettings->algorithm1 != ESTREAM_RANDOM) ||
                (streamnum == 1 && pEstreamSettings->algorithm2 != ESTREAM_RANDOM) ) {
                if (pGlobals->settings->testVectors.saveTestVectors == 1)
                    tvFile  << "(alg n." << ((streamnum==0)?pEstreamSettings->algorithm1:pEstreamSettings->algorithm2) << " - " << ((streamnum==0)?pEstreamSettings->alg1RoundsCount:pEstreamSettings->alg2RoundsCount) << " rounds): ";

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
                        if (pGlobals->settings->testVectors.saveTestVectors) {
                            tvFile << "### ERROR: PLAINTEXT-ENCDECTEXT MISMATCH!" << endl;
                        }
                        status = STAT_PROJECT_ERROR;
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
        mainLogger.out() << "error: unknown testVectorEstreamMethod (" << pEstreamSettings->estreamUsageType << ") in " << shortDescription() << endl;
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
        if (streamnum == 0) tvg = pEstreamSettings->algorithm1;
        else tvg = pEstreamSettings->algorithm2;
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

    return status;
}

int EstreamProject::generateCipherDataStream() {
    if (pEstreamSettings->algorithm1 == ESTREAM_RANDOM) {
        mainLogger.out() << "error: Cannot generate stream from random, cipher must be specified." << endl;
        return STAT_INCOMPATIBLE_PARAMETER;
    } else {
        mainLogger.out() << "info: Generating stream for " << estreamToString(pEstreamSettings->algorithm1);
        if (!pEstreamSettings->limitAlgRounds) {
            mainLogger.out() << " (unlimitted version)" << endl;
        } else {
            mainLogger.out() << " (" << pEstreamSettings->alg1RoundsCount << " rounds)" << endl;
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

    unsigned long alreadyGenerated = 0;
    while (pEstreamSettings->streamSize == 0 ? true : alreadyGenerated < pEstreamSettings->streamSize) {
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
            alreadyGenerated += pGlobals->settings->testVectors.testVectorLength;
        }
    }
    mainLogger.out() << "info: Cipher data stream generation ended. (" << alreadyGenerated << ")" << endl;

    return STAT_OK;
}
