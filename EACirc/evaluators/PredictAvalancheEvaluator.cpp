#include "PredictAvalancheEvaluator.h"
#include "projects/estream/EncryptorDecryptor.h"

AvalancheEvaluator::AvalancheEvaluator()
    : IEvaluator(EVALUATOR_AVALANCHE) { }

void AvalancheEvaluator::evaluateCircuit(unsigned char* outputs, unsigned char* correctOutputs, unsigned char* usePredictorsMask, int* pMatch, int* pTotalPredictCount, int* predictorMatch = NULL){
    unsigned char* inputStream = new unsigned char[pGlobals->settings->testVectors.testVectorLength];
    unsigned char* inputStreamCheck = new unsigned char[pGlobals->settings->testVectors.testVectorLength];
    unsigned char* outputStream = new unsigned char[pGlobals->settings->testVectors.testVectorLength];
    unsigned char* outputStreamCheck = new unsigned char[pGlobals->settings->testVectors.testVectorLength];

    // EDITED when creating infrastructure for projects
    // edit BEGIN
    EncryptorDecryptor* encryptorDecryptor = new EncryptorDecryptor;
    // edit END

	// OUTPUT LAYER CONTAINS CHANGED TV TO ENCRYPT
	encryptorDecryptor->encrypt(correctOutputs,inputStream,0);
	/*encryptorDecryptor->decrypt(inputStream,inputStreamCheck,2);

    for (int input = 0; input < pGACirc->settings->testVectors.testVectorLength; input++) {
		if (correctOutputs[input] != inputStreamCheck[input]) {
            ofstream fitfile(FILE_FITNESS_PROGRESS, ios::app);
            fitfile << "Error! Decrypted text doesn't match the input. See " << FILE_TEST_VECTORS << " for details." << endl;
			fitfile.close();
			exit(1);
		}
	}*/

	// SAVE THE STREAM
	/*if (pGACirc->saveTestVectors == 1) {
		ofstream itvfile("TestData3.txt", ios::app | ios::binary);
        for (int input = 0; input < pGACirc->settings->testVectors.testVectorLength; input++) {
				itvfile << inputStream[input];
		}
		itvfile.close();
	}*/
	
	encryptorDecryptor->encrypt(outputs,outputStream,2);
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

    // EDITED when creating infrastructure for projects
    // edit BEGIN
    delete encryptorDecryptor;
    // edit END
}

string AvalancheEvaluator::shortDescription() {
    return "No description yet.";
}
