#ifndef CIRCUIT_EVALUATOR_INTERFACE_H
#define CIRCUIT_EVALUATOR_INTERFACE_H

#include "EACglobals.h"

class ICircuitEvaluator {
	public:
		ICircuitEvaluator();
        virtual ~ICircuitEvaluator() {}
		ICircuitEvaluator* getCircEvalClass(void);
        virtual void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int* = NULL) {}
};

#endif
