#include "DistinguishTwoEvaluator.h"

DistinguishTwoEvaluator::DistinguishTwoEvaluator() : ICircuitEvaluator(){
}

void DistinguishTwoEvaluator::evaluateCircuit(unsigned char* outputs, unsigned char* correctOutputs, unsigned char* usePredictorsMask, int* pMatch, int* pTotalPredictCount, int* predictorMatch = NULL){
	// OUTPUT LAYER ENCODES 2 DIFFERENT TYPES - 0xff (0x7f-0xff) and 0x00 (0x00-0x7e)
	for (int out = 0; out < pGACirc->sizeOutputLayer; out++) {
		if (((correctOutputs[out] == 0xff) && (outputs[out] > UCHAR_MAX/2)) ||
			((correctOutputs[out] == 0x00) && (outputs[out] <= UCHAR_MAX/2))) {
				(*pMatch)++;
				//if (correctOutputs[out] == 0x00) (*pMatch)++;
				if (predictorMatch != NULL) (predictorMatch[out])++;
				//cout << (int)correctOutputs[out] << ":" << (int)outputs[out] << endl;
		}
		(*pTotalPredictCount)++;
	}
}
