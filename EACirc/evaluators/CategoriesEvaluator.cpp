#include "CategoriesEvaluator.h"

CategoriesEvaluator::CategoriesEvaluator()
    : IEvaluator(EVALUATOR_CATEGORIES) {
    // allocation
    resetEvaluator();
}

CategoriesEvaluator::~CategoriesEvaluator() {
    //deallocation
}

void CategoriesEvaluator::evaluateCircuit(unsigned char* circuitOutputs, unsigned char* referenceOutputs) {
    // update map
}

double CategoriesEvaluator::getFitness() {
    // average euclidean distance
}

void CategoriesEvaluator::resetEvaluator() {
    // reset map
}

string CategoriesEvaluator::shortDescription() {
    return "modular categories evaluator";
}
