#ifndef EVALUATOR_INTERFACE_H
#define EVALUATOR_INTERFACE_H

#include "EACglobals.h"

class IEvaluator {
protected:
    int m_type;
public:
    /** general evaluator constructor
      * - sets evaluator constant
      */
    IEvaluator(int type);

    /** general evaluator destructor
      */
    virtual ~IEvaluator() {}

    /** compares expected circuit outputs to actual ones, updates internal state
      * @param circuitOutputs       output bytes provided by circuit
      * @param referenceOutputs     reference output bytes ("correct output information")
      */
    virtual void evaluateCircuit(unsigned char* circuitOutputs, unsigned char* referenceOutputs) = 0;

    /**
     * return the fitness of the currently inspected individual
     * @return fitness in <0,1>
     */
    virtual float getFitness() = 0;

    /**
     * reset internal state of evaluator (clean stats for next individual)
     */
    virtual void resetEvaluator() = 0;

    /** return short, human readable description of the evaluator
      * @return description
      */
    virtual string shortDescription() = 0;

    /** allocates main evaluator
      * must be called AFTER project initialization
      * @param type     evaluator constant (not project specific evaluator)
      * @return pointer to evaluator instance
      */
    static IEvaluator* getStandardEvaluator(int type);
};

#endif
