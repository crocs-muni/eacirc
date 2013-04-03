#ifndef PREDICT_BITGROUP_PARITY_CIRCUIT_EVALUATOR_H
#define PREDICT_BITGROUP_PARITY_CIRCUIT_EVALUATOR_H

#include "IEvaluator.h"

class PredictBitGroupParityCircuitEvaluator: public IEvaluator {
    unsigned char* usedBits;
public:
    PredictBitGroupParityCircuitEvaluator();
    ~PredictBitGroupParityCircuitEvaluator();
    void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
    string shortDescription();
};

#endif
