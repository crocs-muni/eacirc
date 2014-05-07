/* 
 * File:   FeatureEvaluator.h
 * Author: ph4r05
 *
 * Created on May 7, 2014, 9:23 AM
 */

#ifndef FEATUREEVALUATOR_H
#define	FEATUREEVALUATOR_H

#include "IEvaluator.h"
#include <sstream>

class FeatureEvaluator : public IEvaluator {
public:
    typedef unsigned int featureEvalType;

private:
    featureEvalType * m_categoriesStream0;
    featureEvalType * m_categoriesStream1;
    int m_totalStream0;
    int m_totalStream1;
    
    /** Histogram file buffer to amortize IO requests.
     */
    std::stringstream * m_histBuffer;
    unsigned int * m_histEntries;
    const unsigned int m_histEntriesFlushLimit = 500;
    
    /** Dumps histBuffer to the histogram file, clears the histogram buffer.
     */
    void dumpHistogramFile() const;
public:
    /**
     * allocate categories map according to needed number of categories
     * reset evaluator
     */
    FeatureEvaluator();

    /**
     * deallocated categories maps
     */
    ~FeatureEvaluator();

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
    float getFitness() const;

    /**
     * reset all map fields to zero
     */
    void resetEvaluator();

    /**
     * circuit output bytes are mapped into categories (value modulo number of categories)
     * each output byte is considered separately
     * streams are distinguished by top bit of first byte in reference output
     * fitness is based on weighed Euclidean distance of corresponding categories
     * @return description
     */
    string shortDescription() const;
};


#endif	/* FEATUREEVALUATOR_H */

