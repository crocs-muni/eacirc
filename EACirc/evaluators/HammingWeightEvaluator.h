#ifndef HAMMINGWEIGHTEVALUATOR_H
#define HAMMINGWEIGHTEVALUATOR_H

#include "IEvaluator.h"

class HammingWeightEvaluator : public IEvaluator {
    int* m_weightsStream0;
    int* m_weightsStream1;
    int m_totalStream0;
    int m_totalStream1;
public:
    /**
     * allocate weights map according to output length (number of circuit output bytes * BITS_IN_UCHAR)
     * reset evaluator
     */
    HammingWeightEvaluator();

    /**
     * deallocated weights maps
     */
    ~CategoriesEvaluator();

    /**
     * increase map value for particular output Hamming weight
     * @param circuitOutputs
     * @param referenceOutputs
     */
    void evaluateCircuit(unsigned char* circuitOutputs, unsigned char* referenceOutputs);

    /**
     * Euclidean distance between corresponding weight categories
     * @return fitness
     */
    double getFitness();

    /**
     * reset all map fields to zero
     */
    void resetEvaluator();

    /**
     * circuit output bytes are mapped into categories according to their Hamming weight
     * all output bytes are considered together
     * streams are distinguished by top bit of first byte in reference output
     * fitness is based on weighed Euclidean distance of corresponding categories
     * @return description
     */
    string shortDescription();
};

#endif // HAMMINGWEIGHTEVALUATOR_H
