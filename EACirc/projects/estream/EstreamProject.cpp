#include "EstreamProject.h"
#include "EstreamConstants.h"
#include "EACirc.h"

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
    pEstreamSettings = &estreamSettings;

    return STAT_OK;
}

int EstreamProject::generateTestVectors() {
    if (encryptorDecryptor) delete encryptorDecryptor;
    encryptorDecryptor = new EncryptorDecryptor;

    // USED FOR BALANCING TEST VECTORS
    this->numstats[0] = 0;
    this->numstats[1] = 0;

    for (int testSet = 0; testSet < pGlobals->settings->testVectors.numTestVectors; testSet++) {
        if (pGlobals->settings->testVectors.saveTestVectors == 1) {
            ofstream tvfile(FILE_TEST_VECTORS, ios::app);
            tvfile << "Testset n." << dec << testSet << endl;
            tvfile.close();
        }

        getTestVector();
        for (int input = 0; input < MAX_INPUTS; input++) {
            pGlobals->testVectors[testSet][input] = inputs[input];
        }
        for (int output = 0; output < pGlobals->settings->circuit.sizeOutputLayer; output++)
            pGlobals->testVectors[testSet][MAX_INPUTS+output] = outputs[output];
    }
    return STAT_OK;
}

void EstreamProject::getTestVector(){

    ofstream tvfile(FILE_TEST_VECTORS, ios::app);

    int streamnum = 0;
    bool error = false;

    u8 plain[MAX_INPUTS];// = new u8[STREAM_BLOCK_SIZE];
    u8 outplain[MAX_INPUTS];// = new u8[STREAM_BLOCK_SIZE];

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
                    tvfile  << "(alg n." << ((streamnum==0)?pEstreamSettings->testVectorEstream:pEstreamSettings->testVectorEstream2) << " - " << ((streamnum==0)?pEstreamSettings->limitAlgRoundsCount:pEstreamSettings->limitAlgRoundsCount2) << " rounds): ";

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
                    mainLogger.out() << "error: unknown input type for " << shortDescription() << endl;
                    return;
                }

                encryptorDecryptor->encrypt(plain,inputs,streamnum);
                encryptorDecryptor->decrypt(inputs,outplain,streamnum+2);

            }
            else { // RANDOM
                if (pGlobals->settings->testVectors.saveTestVectors == 1)
                    tvfile << "(RANDOM INPUT - " << rndGen->shortDescription() << "):";
                for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) {
                    rndGen->getRandomFromInterval(255, &outplain[input]);
                    plain[input] = inputs[input] = outplain[input];
                }
            }

            for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) {
                if (outplain[input] != plain[input]) {
                    ofstream fitfile(FILE_FITNESS_PROGRESS, ios::app);
                    fitfile << "Error! Decrypted text doesn't match the input. See " << FILE_TEST_VECTORS << " for details." << endl;
                    fitfile.close();

                    // SIGNALIZE THE ERROR - WE NEED TO LOG INPUTS/OUTPUTS
                    error = true;
                    //exit(1);
                    break;
                }
            }

            break;

    case ESTREAM_BITS_TO_CHANGE:
        //generate the plaintext
        switch (pEstreamSettings->estreamInputType) {
        case ESTREAM_GENTYPE_ZEROS:
            for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) inputs[input] = 0x00;
            break;
        case ESTREAM_GENTYPE_ONES:
            for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) inputs[input] = 0x01;
            break;
        case ESTREAM_GENTYPE_RANDOM:
            for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) rndGen->getRandomFromInterval(255, &(inputs[input]));
            break;
        case ESTREAM_GENTYPE_BIASRANDOM:
            for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++) biasRndGen->getRandomFromInterval(255, &(inputs[input]));
            break;
        default:
            mainLogger.out() << "error: unknown input type for " << shortDescription() << endl;
            return;
        }

        // WE NEED TO LET EVALUATOR KNOW THE INPUTS
        for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++)
            outputs[input] = inputs[input];

        break;

    default:
        mainLogger.out() << "error: unknown testVectorEstreamMethod in " << shortDescription() << endl;
        assert(false);
        break;
    }

    // SAVE TEST VECTORS IN BINARY FILES
    if (pGlobals->settings->testVectors.saveTestVectors == 1) {
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

        int tvg = 0;
        if (streamnum == 0) tvg = pEstreamSettings->testVectorEstream;
        else tvg = pEstreamSettings->testVectorEstream2;
        tvfile << setfill('0');

        if (memcmp(inputs,plain,pGlobals->settings->testVectors.testVectorLength) != 0) {
            for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++)
            tvfile << setw(2) << hex << (int)(plain[input]);
            tvfile << "::";
        }

        for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++)
            tvfile << setw(2) << hex << (int)(inputs[input]);

        if (memcmp(inputs,outplain,pGlobals->settings->testVectors.testVectorLength) != 0) {
            tvfile << "::";
            for (int input = 0; input < pGlobals->settings->testVectors.testVectorLength; input++)
            tvfile << setw(2) << hex << (int)(outplain[input]);
        }
        tvfile << endl;
    }

    tvfile.close();

    // THERE WAS AN ERROR, EXIT...
    if (error) exit(1);

    return;
}
