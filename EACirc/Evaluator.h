#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "EACglobals.h"
#include "test_vector_generator/ITestVectGener.h"
#include "CircuitGenome.h"

class Evaluator {
		ITestVectGener *TVG;
	public:
		Evaluator();
        ~Evaluator();
		void generateTestVectors();
		int evaluateStep(GA1DArrayGenome<unsigned long> genome, int actGener);
};

#endif
