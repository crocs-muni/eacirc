#ifndef PREDICT_HAMMING_WEIGHT_CIRCUIT_EVALUATOR_H
#define PREDICT_HAMMING_WEIGHT_CIRCUIT_EVALUATOR_H

#include "ICircuitEvaluator.h"

class PredictHammingWeightCircuitEvaluator: public ICircuitEvaluator {
	public:
		PredictHammingWeightCircuitEvaluator();
		void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
};

#endif