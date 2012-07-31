#ifndef CIRCUIT_EVALUATOR_INTERFACE_H
#define CIRCUIT_EVALUATOR_INTERFACE_H

#include "../SSGlobals.h"

class ICircuitEvaluator {
	public:
		ICircuitEvaluator();
		ICircuitEvaluator* getCircEvalClass(void);
		virtual void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int* = NULL) {};
};

#endif