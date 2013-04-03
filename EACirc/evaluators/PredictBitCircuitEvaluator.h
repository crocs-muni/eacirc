#ifndef PREDICT_BIT_CIRCUIT_EVALUATOR_H
#define PREDICT_BIT_CIRCUIT_EVALUATOR_H

#include "IEvaluator.h"

class PredictBitCircuitEvaluator: public IEvaluator {
public:
    PredictBitCircuitEvaluator();
    void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
    string shortDescription();
};

#endif
