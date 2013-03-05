#ifndef EVALUATOR_INTERFACE_H
#define EVALUATOR_INTERFACE_H

#include "EACglobals.h"

class IEvaluator {
protected:
    int m_type;
public:
    IEvaluator(int type);
    virtual ~IEvaluator() {}
    virtual void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int* = NULL) = 0;
    virtual string shortDescription() = 0;

    static IEvaluator* getEvaluator(int type);
};

#endif
