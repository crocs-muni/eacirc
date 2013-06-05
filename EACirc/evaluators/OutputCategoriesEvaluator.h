#ifndef OUTPUT_CATEGORIES_EVALUATOR_H
#define OUTPUT_CATEGORIES_EVALUATOR_H

#include "IEvaluator.h"

class OutputCategoriesEvaluator: public IEvaluator {
public:
    OutputCategoriesEvaluator();
    void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
    string shortDescription();
};

#endif