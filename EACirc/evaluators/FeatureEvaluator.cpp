/*
 * File:   FeatureEvaluator.cpp
 * Author: ph4r05
 *
 * Created on May 7, 2014, 9:23 AM
 */

#include "FeatureEvaluator.h"
#include "CommonFnc.h"

#include <cassert>
#define BITS_IN_OUTPUT (8u*sizeof(unsigned char))

FeatureEvaluator::FeatureEvaluator() : IEvaluator(EVALUATOR_CATEGORIES),
      m_categoriesStream0(NULL), m_categoriesStream1(NULL),
      m_totalStream0(0), m_totalStream1(0),
      m_histBuffer(NULL), m_histEntries(NULL) {
    
    m_categoriesStream0 = new featureEvalType[pGlobals->settings->main.evaluatorPrecision];
    m_categoriesStream1 = new featureEvalType[pGlobals->settings->main.evaluatorPrecision];
    
    m_histBuffer = new std::stringstream;
    m_histEntries = new unsigned int;
    (*m_histEntries) = 0;
    resetEvaluator();
}

FeatureEvaluator::~FeatureEvaluator() {
    if (m_categoriesStream0!=NULL){
        delete m_categoriesStream0;
    }
    m_categoriesStream0 = NULL;
    
    if (m_categoriesStream1!=NULL){
        delete m_categoriesStream1;
    }
    m_categoriesStream1 = NULL;
    
    dumpHistogramFile();
    if (m_histBuffer != NULL){
        delete m_histBuffer;
    }
    m_histBuffer = NULL;
    
    if (m_histEntries!=NULL){
        delete m_histEntries;
    }
    m_histEntries = NULL;
}

void FeatureEvaluator::dumpHistogramFile() const {
    // Only if we have something to dump.
    if (m_histBuffer->tellp() <= 0){
        return;
    }
    
    // Fitness to histograms.
    ofstream hist(FILE_HISTOGRAMS, ios_base::app);
    hist << (m_histBuffer->str());
    m_histBuffer->flush();
    hist.close();
    
    // Clear string buffer.
    m_histBuffer->str(std::string());
    m_histBuffer->clear();
    *m_histEntries = 0;
}

void FeatureEvaluator::evaluateCircuit(unsigned char* circuitOutputs, unsigned char* referenceOutputs) {
    // select stream map to update and increase total in corresponding counter
    featureEvalType * currentStreamMap;
    // Highest bit in referenceOutputs[0] determines which data was used
    // in the evaluation of the circuit (i.e., from algorithm 1 or algorithm 2).
    if (referenceOutputs[0] >> (BITS_IN_UCHAR-1) == 0) {
        currentStreamMap = m_categoriesStream0;
        m_totalStream0++;
    } else {
        currentStreamMap = m_categoriesStream1;
        m_totalStream1++;
    }
    
    assert(pGlobals->settings->main.evaluatorPrecision>0);
    
    // Each bit corresponds to the separate feature of the data set.
    const unsigned int maxOutBits = BITS_IN_OUTPUT*pGlobals->settings->main.circuitSizeOutput;
    for (unsigned int bit = 0; bit < static_cast<unsigned int>(pGlobals->settings->main.evaluatorPrecision) && bit < maxOutBits; bit++){
        // Increment counter for particular feature if it is enabled in the circuit output.
        currentStreamMap[bit] += (circuitOutputs[bit / BITS_IN_OUTPUT] & (1u << (bit % BITS_IN_OUTPUT))) > 0;
    }
}

float FeatureEvaluator::getFitness() const {
    float chiSquareValue = 0;
    int dof = 0;
    // using two-smaple Chi^2 test (http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm)
    float k1 = 1;
    float k2 = 1;
    for (int category = 0; category < pGlobals->settings->main.evaluatorPrecision; category++) {
        if (m_categoriesStream0[category] + m_categoriesStream1[category] > 5) {
            dof++;
            chiSquareValue += pow(k1*m_categoriesStream1[category]-k2*m_categoriesStream0[category], 2) /
                              (m_categoriesStream0[category] + m_categoriesStream1[category]);
        }
    }
    dof--; // last category is fully determined by others
    float fitness = (1.0 - chisqr(dof,chiSquareValue));
    
    // Fitness to histograms.
    (*m_histBuffer) << pGlobals->stats.actGener << endl;
    for (int category = 0; category < pGlobals->settings->main.evaluatorPrecision; category++) {
        (*m_histBuffer) << setw(4) << right << setfill('0') << m_categoriesStream0[category] << " ";
    }
    (*m_histBuffer) << endl;
    
    for (int category = 0; category < pGlobals->settings->main.evaluatorPrecision; category++) {
        (*m_histBuffer) << setw(4) << right << setfill('0') << m_categoriesStream1[category] << " ";
    }
    (*m_histBuffer) << endl << endl;
    (*m_histEntries) += 1;
    
    // If buffer is large enough, dump it to the histogram file.
    if ((*m_histEntries) >= m_histEntriesFlushLimit){
        dumpHistogramFile();
    }
    
    return fitness;
}

void FeatureEvaluator::resetEvaluator() {
    m_totalStream0 = m_totalStream1 = 0;
    memset(m_categoriesStream0, 0, pGlobals->settings->main.evaluatorPrecision * sizeof(featureEvalType));
    memset(m_categoriesStream1, 0, pGlobals->settings->main.evaluatorPrecision * sizeof(featureEvalType));
    
    // If buffer is large enough, dump it to the histogram file.
    if (m_histEntries != NULL && (*m_histEntries) >= m_histEntriesFlushLimit){
        dumpHistogramFile();
    }
}


string FeatureEvaluator::shortDescription() const {
    return "FeatureEvaluator";
}
