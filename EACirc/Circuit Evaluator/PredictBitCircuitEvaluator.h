#ifndef PREDICT_BIT_CIRCUIT_EVALUATOR_H
#define PREDICT_BIT_CIRCUIT_EVALUATOR_H

#include "ICircuitEvaluator.h"

class PredictBitCircuitEvaluator: public ICircuitEvaluator {
	public:
		PredictBitCircuitEvaluator();
		void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
};

#endif