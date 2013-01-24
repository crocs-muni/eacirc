#include "PredictByteCircuitEvaluator.h"

PredictByteCircuitEvaluator::PredictByteCircuitEvaluator() : ICircuitEvaluator(){
}

void PredictByteCircuitEvaluator::evaluateCircuit(unsigned char* outputs, unsigned char* correctOutputs, unsigned char* usePredictorsMask, int* pMatch, int* pTotalPredictCount, int* predictorMatch = NULL){
	// OUTPUT LAYER ENCODES DIRECTLY THE EXPECTED BYTES
    // COUNT NUMBER OF CORRECT BYTES
    for (int out = 0; out < pGACirc->settings->circuit.sizeOutputLayer; out++) {
		if (outputs[out] == correctOutputs[out]) {
			(*pMatch)++;
			if (predictorMatch != NULL) (predictorMatch[0])++;
		}          
		(*pTotalPredictCount)++;                    					
	}
}
