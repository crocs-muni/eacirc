#ifndef DISTINGUISH_TWO_EVALUATOR_H
#define DISTINGUISH_TWO_EVALUATOR_H

#include "IEvaluator.h"

class DistinguishTwoEvaluator: public IEvaluator {
public:
    DistinguishTwoEvaluator();
    void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
    string shortDescription();
};

#endif
