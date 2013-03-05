#ifndef PREDICT_BYTE_CIRCUIT_EVALUATOR_H
#define PREDICT_BYTE_CIRCUIT_EVALUATOR_H

#include "IEvaluator.h"

class PredictByteCircuitEvaluator: public IEvaluator {
public:
    PredictByteCircuitEvaluator();
    void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
    string shortDescription();
};

#endif
