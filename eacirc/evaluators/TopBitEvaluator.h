#ifndef TOPBITEVALUATOR_H
#define TOPBITEVALUATOR_H

#include "IEvaluator.h"

class TopBitEvaluator: public IEvaluator {
    int m_matchedOutputBytes;
    int m_totalOutputBytes;
public:
    /**
     * reset initial state
     */
    TopBitEvaluator();

    /**
     * increase total output byte count
     * check top bit and increase matched output byte cout if necessary
     * only top bit of 1st reference byte is considered
     * @param circuitOutputs
     * @param referenceOutputs
     */
    virtual void evaluateCircuit(unsigned char* circuitOutputs, unsigned char* referenceOutputs);

    /**
     * fitness is computed as quotient of successfully matched and all results
     * @return fitness
     */
    virtual float getFitness() const;

    /**
     * reset counters to zero
     */
    virtual void resetEvaluator();

    /**
     * evaluator distinguishes vectors by the top bit of output bytes
     * each output byte is considered separately
     * @return description
     */
    virtual string shortDescription() const;
};

#endif // TOPBITEVALUATOR_H
