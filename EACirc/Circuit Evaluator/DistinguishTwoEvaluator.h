#ifndef DISTINGUISH_TWO_EVALUATOR_H
#define DISTINGUISH_TWO_EVALUATOR_H

#include "ICircuitEvaluator.h"

class DistinguishTwoEvaluator: public ICircuitEvaluator {
	public:
		DistinguishTwoEvaluator();
		void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
};

#endif