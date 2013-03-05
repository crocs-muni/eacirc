#ifndef PREDICT_HAMMING_WEIGHT_CIRCUIT_EVALUATOR_H
#define PREDICT_HAMMING_WEIGHT_CIRCUIT_EVALUATOR_H

#include "IEvaluator.h"

class PredictHammingWeightCircuitEvaluator: public IEvaluator {
public:
    PredictHammingWeightCircuitEvaluator();
    void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
    string shortDescription();
};

#endif
