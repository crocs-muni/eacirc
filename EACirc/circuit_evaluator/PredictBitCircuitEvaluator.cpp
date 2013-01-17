#include "PredictBitCircuitEvaluator.h"

PredictBitCircuitEvaluator::PredictBitCircuitEvaluator() : ICircuitEvaluator(){
}

void PredictBitCircuitEvaluator::evaluateCircuit(unsigned char* outputs, unsigned char* correctOutputs, unsigned char* usePredictorsMask, int* pMatch, int* pTotalPredictCount, int* predictorMatch = NULL){
	// OUTPUT LAYER ENCODES DIRECTLY THE EXPECTED BITS
	// COUNT NUMBER OF CORRECT BITS 
	for (int out = 0; out < pGACirc->sizeOutputLayer; out++) {
		for (int bit = 0; bit < NUM_BITS; bit++) {
			if (usePredictorsMask[out] == 1) {
				// COMPARE VALUE ON bit-th POSITION
				if ((outputs[out] & (unsigned char) pGACirc->precompPow[bit]) == (correctOutputs[out] & (unsigned char) pGACirc->precompPow[bit])) {
					(*pMatch)++;
					if (predictorMatch != NULL) (predictorMatch[out])++;
				}
	                    
				(*pTotalPredictCount)++;                    
			}
		}
	}
}
