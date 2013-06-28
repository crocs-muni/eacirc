#include "HammingWeightEvaluator.h"

HammingWeightEvaluator::HammingWeightEvaluator()
    : IEvaluator(EVALUATOR_HAMMING_WEIGHT), m_weightsStream0(NULL), m_weightsStream1(NULL),
      m_totalStream0(0), m_totalStream1(0) {
    m_weightsStream0 = new int[pGlobals->settings->circuit.sizeOutputLayer * BITS_IN_UCHAR];
    m_weightsStream1 = new int[pGlobals->settings->circuit.sizeOutputLayer * BITS_IN_UCHAR];
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
    for (int outputByte = 0; outputByte < pGlobals->settings->circuit.sizeOutputLayer; outputByte++) {
        for (int bit = 0; bit < BITS_IN_UCHAR; bit++) {
            if (circuitOutputs[outputByte] & (unsigned char) pGlobals->precompPow[bit]) hammingWeight++;
        }
    }
    currentStreamMap[hammingWeight]++;
}

float HammingWeightEvaluator::getFitness() {
    float fitness = 0;
    float temp0, temp1;
    // add normalised Euclidean distance for each category
    for (int category = 0; category < pGlobals->settings->circuit.sizeOutputLayer * BITS_IN_UCHAR; category++) {
        temp0 = m_totalStream0 ? m_weightsStream0[category] / (float) m_totalStream0 : 0;
        temp1 = m_totalStream1 ? m_weightsStream1[category] / (float) m_totalStream1 : 0;
        fitness += pow(temp0 - temp1, 2);
    }
    // transform fitness from interval <0,2> to interval <0,1>
    fitness = fitness / 2;
    return fitness;
}

void HammingWeightEvaluator::resetEvaluator() {
    m_totalStream0 = m_totalStream1 = 0;
    memset(m_weightsStream0, 0, pGlobals->settings->circuit.sizeOutputLayer * BITS_IN_UCHAR * sizeof(int));
    memset(m_weightsStream1, 0, pGlobals->settings->circuit.sizeOutputLayer * BITS_IN_UCHAR * sizeof(int));
}

string HammingWeightEvaluator::shortDescription() {
    return "Hamming weight categories evaluator";
}
