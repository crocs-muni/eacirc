#ifndef CATEGORIESEVALUATOR_H
#define CATEGORIESEVALUATOR_H

#include "IEvaluator.h"
#define CATEGORY_WARNING_THRESHOLD 0

class CategoriesEvaluator : public IEvaluator {
    int* m_categoriesStream0;
    int* m_categoriesStream1;
public:
    /**
     * allocate categories map according to needed number of categories
     * reset evaluator
     */
    CategoriesEvaluator();

    /**
     * deallocated categories maps
     */
    ~CategoriesEvaluator();

    /**
     * increase map value for output byte value modulo number of categories
     * (for each output byte)
     * @param circuitOutputs
     * @param referenceOutputs
     */
    void evaluateCircuit(unsigned char* circuitOutputs, unsigned char* referenceOutputs);

    /**
     * Euclidean distance between corresponding output categories
     * @return fitness
     */
    double getFitness();

    /**
     * reset all map fields to zero
     */
    void resetEvaluator();

    /**
     * circuit output bytes are mapped into categories (value modulo number of categories)
     * each output byte is considered separately
     * fitness is based on weighed Euclidean distance of corresponding categories
     * @return description
     */
    string shortDescription();
};

#endif // CATEGORIESEVALUATOR_H
