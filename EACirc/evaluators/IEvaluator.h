#ifndef EVALUATOR_INTERFACE_H
#define EVALUATOR_INTERFACE_H

#include "EACglobals.h"

class IEvaluator {
protected:
    int m_type;
public:
    /** general evaluator constructor
      * sets evaluator constant
      */
    IEvaluator(int type);

    /** general evaluator destructor
      */
    virtual ~IEvaluator() {}

    /** compares expected circuit outputs to actual ones
      * TBD/TODO
      */
    virtual void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int* = NULL) = 0;

    /** return short, human readable description of the evaluator
      * @return description
      */
    virtual string shortDescription() = 0;

    /** allocates main evaluator to globally accessible resources
      * must be called AFTER project initialization
      * @param type     evaluator constant (not project specific evaluator)
      * @return pointer to evaluator instance
      */
    static IEvaluator* getStandardEvaluator(int type);
};

#endif
