#include "PredictHammingWeightCircuitEvaluator.h"

PredictHammingWeightCircuitEvaluator::PredictHammingWeightCircuitEvaluator() : ICircuitEvaluator(){
}

void PredictHammingWeightCircuitEvaluator::evaluateCircuit(unsigned char* outputs, unsigned char* correctOutputs, unsigned char* usePredictorsMask, int* pMatch, int* pTotalPredictCount, int* predictorMatch = NULL){
	// HAMMING WEIGHT OF BYTES IN OUTPUT LAYER SHOULD MATCH WEIGHT IN CORRECT OUTPUT
    int predictWeight = 0;
    int correctWeight = 0;
            
    for (int out = 0; out < pGACirc->outputLayerSize; out++) {
		if (usePredictorsMask[out] == 1) {
			predictWeight = 0;
			correctWeight = 0;
			// GET PREDICTION OF WEIGHT 
			for (int bit = 0; bit < NUM_BITS; bit++) {
				if (outputs[out] & (unsigned char) pGACirc->precompPow[bit]) predictWeight++;
			}
	                
			// GET REAL WEIGHT 
			for (int bit = 0; bit < NUM_BITS; bit++) {
				if (correctOutputs[out] & (unsigned char) pGACirc->precompPow[bit]) correctWeight++;
			}

			// OBTAIN POINTS FOR CORRECTNESS - IF COMPLETELY MATCH, THAN MAXIMUM POINTS ARE OBTAINED
			int points = NUM_BITS - abs(predictWeight - correctWeight);
			*pMatch += points;
			if (predictorMatch != NULL) (predictorMatch[out])+= points;
			(*pTotalPredictCount)++;         

	                
			// CHECK IF MATCH
			if (predictWeight == correctWeight) {
				(*pMatch)++;
				if (predictorMatch != NULL) (predictorMatch[out])++;
			}
			(*pTotalPredictCount)++;         
/**/
		}
	}
            
}
