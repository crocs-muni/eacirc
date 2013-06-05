#include "OutputCategoriesEvaluator.h"

OutputCategoriesEvaluator::OutputCategoriesEvaluator()
    : IEvaluator(EVALUATOR_OUTPUT_CATEGORIES) { }

void OutputCategoriesEvaluator::evaluateCircuit(unsigned char* outputs, unsigned char* correctOutputs, unsigned char* usePredictorsMask, int* pMatch, int* pTotalPredictCount, int* predictorMatch = NULL){
	// Compute Euclidean distance between circuitOutputCategories for given inputs and for truly random inputs
	double	difference = 0;
	for (int i = 0; i < NUM_OUTPUT_CATEGORIES; i++) {
		difference += pow(pGlobals->testVectors.circuitOutputCategories[i] - pGlobals->testVectors.circuitOutputCategoriesRandom[i], 2);
	}
	
	// Fitness idea: if random input is presented, circuit should provide as low difference as possible (difference)
	// If function input is presented, circuit should provide as large difference as possible (difference)
	if (correctOutputs[0] == 0x00) *pMatch += difference; 
	if (correctOutputs[0] == 0xff) *pMatch -= difference; 
}

string OutputCategoriesEvaluator::shortDescription() {
    return "Compare histogram of output categories for given inputs and truly random inputs.";
}
