#include "AvalancheEvaluator.h"
#include "EncryptorDecryptor.h"

AvalancheEvaluator::AvalancheEvaluator(EncryptorDecryptor* encryptorDecryptor)
    : IEvaluator(ESTREAM_EVALUATOR_AVALANCHE), m_encryptorDecryptor(encryptorDecryptor) { }

void AvalancheEvaluator::evaluateCircuit(unsigned char* outputs, unsigned char* correctOutputs, unsigned char* usePredictorsMask, int* pMatch, int* pTotalPredictCount, int* predictorMatch = NULL){
    unsigned char* inputStream = new unsigned char[pGlobals->settings->testVectors.inputLength];
    unsigned char* inputStreamCheck = new unsigned char[pGlobals->settings->testVectors.inputLength];
    unsigned char* outputStream = new unsigned char[pGlobals->settings->testVectors.inputLength];
    unsigned char* outputStreamCheck = new unsigned char[pGlobals->settings->testVectors.inputLength];
    unsigned char* ignoreStream = new unsigned char[pGlobals->settings->testVectors.inputLength];

    // OUTPUT LAYER CONTAINS CHANGED TV TO ENCRYPT
    m_encryptorDecryptor->encrypt(correctOutputs,inputStream,0,0);
    // added later: to ensure cipher streams integrity
    m_encryptorDecryptor->encrypt(correctOutputs,ignoreStream,1,0);

    /*encryptorDecryptor->decrypt(inputStream,inputStreamCheck,2);

    for (int input = 0; input < pGACirc->settings->testVectors.testVectorLength; input++) {
        if (correctOutputs[input] != inputStreamCheck[input]) {
            ofstream fitfile(FILE_FITNESS_PROGRESS, ios::app);
            fitfile << "Error! Decrypted text doesn't match the input. See " << FILE_TEST_VECTORS << " for details." << endl;
            fitfile.close();
            exit(1);
        }
    }

    // SAVE THE STREAM
    if (pGACirc->saveTestVectors == 1) {
        ofstream itvfile("TestData3.txt", ios::app | ios::binary);
        for (int input = 0; input < pGACirc->settings->testVectors.testVectorLength; input++) {
                itvfile << inputStream[input];
        }
        itvfile.close();
    }*/

    m_encryptorDecryptor->encrypt(outputs,outputStream,0,1);
    // added later: to ensure cipher streams integrity
    m_encryptorDecryptor->encrypt(outputs,ignoreStream,1,1);

    /*encryptorDecryptor->decrypt(outputStream,outputStreamCheck,0);

    for (int input = 0; input < pGACirc->settings->testVectors.testVectorLength; input++) {
        if (outputs[input] != inputStreamCheck[input]) {
            ofstream fitfile(FILE_FITNESS_PROGRESS, ios::app);
            fitfile << "Error! Decrypted text doesn't match the input. See " << FILE_TEST_VECTORS << " for details." << endl;
            fitfile.close();
            exit(1);
        }
    }*/

    // SAVE THE STREAM
    /*if (pGACirc->saveTestVectors == 1) {
        ofstream itvfile("TestData4.txt", ios::app | ios::binary);
        for (int input = 0; input < pGACirc->settings->testVectors.testVectorLength; input++) {
            itvfile << outputStream[input];
        }
        itvfile.close();
    }*/

    int ppMatch = 0;

    // COMPARE THE STREAMS
    for (int out = 0; out < pGlobals->settings->circuit.sizeOutputLayer; out++) {
        for (int bit = 0; bit < BITS_IN_UCHAR; bit++) {
            // COMPARE VALUE ON bit-th POSITION
            if ((inputStream[out] & (unsigned char) pGlobals->precompPow[bit]) == (outputStream[out] & (unsigned char) pGlobals->precompPow[bit]))
                ppMatch++;
            else
                ppMatch--;
            (*pTotalPredictCount) ++;
        }
    }

    (*pMatch) += abs(ppMatch);

    delete[] inputStream;
    delete[] outputStream;
    delete[] inputStreamCheck;
    delete[] outputStreamCheck;
    delete[] ignoreStream;
}

string AvalancheEvaluator::shortDescription() {
    return "eStream project: evaluator for based on strict avalanche criterion";
}
