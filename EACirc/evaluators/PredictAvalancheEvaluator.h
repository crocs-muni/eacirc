#ifndef AVALANCHE_EVALUATOR_H
#define AVALANCHE_EVALUATOR_H

#include "IEvaluator.h"

class AvalancheEvaluator: public IEvaluator {
public:
    AvalancheEvaluator();
    void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
    string shortDescription();
};

#endif
