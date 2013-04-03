#include "PredictBytesParityCircuitEvaluator.h"

PredictBytesParityCircuitEvaluator::PredictBytesParityCircuitEvaluator()
    : IEvaluator(EVALUATOR_BYTES_PARITY) { }

void PredictBytesParityCircuitEvaluator::evaluateCircuit(unsigned char* outputs, unsigned char* correctOutputs, unsigned char* usePredictorsMask, int* pMatch, int* pTotalPredictCount, int* predictorMatch = NULL){
	// PARITY OF BYTES IN OUTPUT LAYER SHOULD MATCH PARITY IN CORRECT OUTPUT
    int predictParity = 0;
    int correctParity = 0;
            
    for (int out = 0; out < pGlobals->settings->circuit.sizeOutputLayer; out++) {
		if (usePredictorsMask[out] == 1) {
			predictParity = 0;
			correctParity = 0;
			// GET PREDICTION OF PARITY, if hamming weight > BITS_IN_UCHAR/2 then parity = 0, otherwise pariy = 1
			for (int bit = 0; bit < BITS_IN_UCHAR; bit++) {
				if (outputs[out] & (unsigned char) pGlobals->precompPow[bit]) predictParity++;
			}
			if (predictParity > (BITS_IN_UCHAR / 2)) predictParity = 0;
			else predictParity = 1;
	                
			// GET REAL PARITY 
			for (int bit = 0; bit < BITS_IN_UCHAR; bit++) {
				if (correctOutputs[out] & (unsigned char) pGlobals->precompPow[bit]) correctParity++;
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

string PredictBytesParityCircuitEvaluator::shortDescription() {
    return "No description yet.";
}
