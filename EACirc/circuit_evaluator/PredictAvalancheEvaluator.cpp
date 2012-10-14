#include "PredictAvalancheEvaluator.h"
#include "estream/EncryptorDecryptor.h"

AvalancheEvaluator::AvalancheEvaluator() : ICircuitEvaluator(){
}

void AvalancheEvaluator::evaluateCircuit(unsigned char* outputs, unsigned char* correctOutputs, unsigned char* usePredictorsMask, int* pMatch, int* pTotalPredictCount, int* predictorMatch = NULL){
	unsigned char* inputStream = new unsigned char[pGACirc->testVectorLength];
	unsigned char* inputStreamCheck = new unsigned char[pGACirc->testVectorLength];
	unsigned char* outputStream = new unsigned char[pGACirc->testVectorLength];
	unsigned char* outputStreamCheck = new unsigned char[pGACirc->testVectorLength];

	// OUTPUT LAYER CONTAINS CHANGED TV TO ENCRYPT
	encryptorDecryptor->encrypt(correctOutputs,inputStream,0);
	/*encryptorDecryptor->decrypt(inputStream,inputStreamCheck,2);

	for (int input = 0; input < pGACirc->testVectorLength; input++) {
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
		for (int input = 0; input < pGACirc->testVectorLength; input++) {
				itvfile << inputStream[input];
		}
		itvfile.close();
	}*/
	
	encryptorDecryptor->encrypt(outputs,outputStream,2);
	/*encryptorDecryptor->decrypt(outputStream,outputStreamCheck,0);

	for (int input = 0; input < pGACirc->testVectorLength; input++) {
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
		for (int input = 0; input < pGACirc->testVectorLength; input++) {
			itvfile << outputStream[input];
		}
		itvfile.close();
	}*/
	
	int ppMatch = 0;

	// COMPARE THE STREAMS
	for (int out = 0; out < pGACirc->outputLayerSize; out++) {
		for (int bit = 0; bit < NUM_BITS; bit++) {
			// COMPARE VALUE ON bit-th POSITION
			if ((inputStream[out] & (unsigned char) pGACirc->precompPow[bit]) == (outputStream[out] & (unsigned char) pGACirc->precompPow[bit]))
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
}
