#include "HammingWeightEvaluator.h"

#define max(a,b) (((a)>(b))?(a):(b))

HammingWeightEvaluator::HammingWeightEvaluator()
    : IEvaluator(EVALUATOR_HAMMING_WEIGHT), m_weightsStream0(NULL), m_weightsStream1(NULL),
      m_totalStream0(0), m_totalStream1(0), m_numUnderThreshold(0) {
    m_weightsStream0 = new int[pGlobals->settings->main.circuitSizeOutput * BITS_IN_UCHAR + 1];
    m_weightsStream1 = new int[pGlobals->settings->main.circuitSizeOutput * BITS_IN_UCHAR + 1];
    resetEvaluator();
}

HammingWeightEvaluator::~HammingWeightEvaluator() {
    delete m_weightsStream0;
    m_weightsStream0 = NULL;
    delete m_weightsStream1;
    m_weightsStream1 = NULL;
}

void HammingWeightEvaluator::evaluateCircuit(unsigned char* circuitOutputs, unsigned char* referenceOutputs) {
    // select stream map to update and increase total in corresponding counter
    int* currentStreamMap;
    if (referenceOutputs[0] >> (BITS_IN_UCHAR-1) == 0) {
        currentStreamMap = m_weightsStream0;
        m_totalStream0++;
    } else {
        currentStreamMap = m_weightsStream1;
        m_totalStream1++;
    }
    // compute Hamming weight of the circuit output
    int hammingWeight = 0;
    for (int outputByte = 0; outputByte < pGlobals->settings->main.circuitSizeOutput; outputByte++) {
        for (int bit = 0; bit < BITS_IN_UCHAR; bit++) {
            if (circuitOutputs[outputByte] & (unsigned char) pGlobals->precompPow[bit]) hammingWeight++;
        }
    }
    currentStreamMap[hammingWeight]++;
}

float HammingWeightEvaluator::getFitness() const {
    float fitness = 0;
    // compute Pearson's Chi square test
    // chi^2 = sum_{i=1}^{n}{\frac{(Observed_i-Expected_i)^2}{Expected_i}}
    // check for threshold E_i >=5, Q_i >=5
    for (int category = 0; category < pGlobals->settings->main.circuitSizeOutput * BITS_IN_UCHAR + 1; category++) {
        float divider = max(m_weightsStream0[category], 1); // prevent division by zero
        fitness += pow(m_weightsStream1[category]-m_weightsStream0[category], 2) / divider;
    }
    return fitness;
}

void HammingWeightEvaluator::resetEvaluator() {
    unsigned long tempNumUnderThreshold = 0;
    for (int category = 0; category < pGlobals->settings->main.evaluatorPrecision; category++) {
        if (m_weightsStream0[category] < CATEGORY_THRESHOLD) tempNumUnderThreshold++;
        if (m_weightsStream1[category] < CATEGORY_THRESHOLD) tempNumUnderThreshold++;
    }
    if (tempNumUnderThreshold != 0) m_numUnderThreshold += tempNumUnderThreshold;
    m_totalStream0 = m_totalStream1 = 0;
    memset(m_weightsStream0, 0, (pGlobals->settings->main.circuitSizeOutput * BITS_IN_UCHAR + 1) * sizeof(int));
    memset(m_weightsStream1, 0, (pGlobals->settings->main.circuitSizeOutput * BITS_IN_UCHAR + 1) * sizeof(int));
}

string HammingWeightEvaluator::shortDescription() const {
    return "Hamming weight categories evaluator";
}
