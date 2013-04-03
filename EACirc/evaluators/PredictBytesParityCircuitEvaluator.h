#ifndef PREDICT_BYTES_PARITY_CIRCUIT_EVALUATOR_H
#define PREDICT_BYTES_PARITY_CIRCUIT_EVALUATOR_H

#include "IEvaluator.h"

class PredictBytesParityCircuitEvaluator: public IEvaluator {
public:
    PredictBytesParityCircuitEvaluator();
    void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
    string shortDescription();
};

#endif
