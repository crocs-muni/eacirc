#ifndef AVALANCHE_EVALUATOR_H
#define AVALANCHE_EVALUATOR_H

#include "ICircuitEvaluator.h"

class AvalancheEvaluator: public ICircuitEvaluator {
	public:
		AvalancheEvaluator();
		void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
};

#endif