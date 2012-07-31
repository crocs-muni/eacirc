#ifndef PREDICT_BYTE_CIRCUIT_EVALUATOR_H
#define PREDICT_BYTE_CIRCUIT_EVALUATOR_H

#include "ICircuitEvaluator.h"

class PredictByteCircuitEvaluator: public ICircuitEvaluator {
	public:
		PredictByteCircuitEvaluator();
		void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
};

#endif