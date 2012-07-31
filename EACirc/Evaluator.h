#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "SSGlobals.h"
#include "ITestVectGener.h"
#include "CircuitGenome.h"

class Evaluator {
		ITestVectGener *TVG;
	public:
		Evaluator();
		void generateTestVectors();
		int evaluateStep(GA1DArrayGenome<unsigned long> genome, int actGener);
};

#endif