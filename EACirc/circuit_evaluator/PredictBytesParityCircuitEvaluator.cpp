#include "PredictBytesParityCircuitEvaluator.h"

PredictBytesParityCircuitEvaluator::PredictBytesParityCircuitEvaluator() : ICircuitEvaluator(){
}

void PredictBytesParityCircuitEvaluator::evaluateCircuit(unsigned char* outputs, unsigned char* correctOutputs, unsigned char* usePredictorsMask, int* pMatch, int* pTotalPredictCount, int* predictorMatch = NULL){
	// PARITY OF BYTES IN OUTPUT LAYER SHOULD MATCH PARITY IN CORRECT OUTPUT
    int predictParity = 0;
    int correctParity = 0;
            
    for (int out = 0; out < pGACirc->sizeOutputLayer; out++) {
		if (usePredictorsMask[out] == 1) {
			predictParity = 0;
			correctParity = 0;
			// GET PREDICTION OF PARITY, if hamming weight > NUM_BITS/2 then parity = 0, otherwise pariy = 1
			for (int bit = 0; bit < NUM_BITS; bit++) {
				if (outputs[out] & (unsigned char) pGACirc->precompPow[bit]) predictParity++;
			}
			if (predictParity > (NUM_BITS / 2)) predictParity = 0;
			else predictParity = 1;
	                
			// GET REAL PARITY 
			for (int bit = 0; bit < NUM_BITS; bit++) {
				if (correctOutputs[out] & (unsigned char) pGACirc->precompPow[bit]) correctParity++;
			}
			correctParity = (correctParity & 0x01) ? 1 : 0; 
	                
			// CHECK IF MATCH
			if (predictParity == correctParity) {
				(*pMatch)++;
				if (predictorMatch != NULL) (predictorMatch[out])++;
			}
			(*pTotalPredictCount)++;                    
		}
    }
}
