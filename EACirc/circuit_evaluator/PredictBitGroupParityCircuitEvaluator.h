#ifndef PREDICT_BITGROUP_PARITY_CIRCUIT_EVALUATOR_H
#define PREDICT_BITGROUP_PARITY_CIRCUIT_EVALUATOR_H

#include "ICircuitEvaluator.h"

class PredictBitGroupParityCircuitEvaluator: public ICircuitEvaluator {
	public:
		PredictBitGroupParityCircuitEvaluator();
		void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
};

#endif