#include "TopBitEvaluator.h"

TopBitEvaluator::TopBitEvaluator()
    : IEvaluator(EVALUATOR_TOP_BIT), m_matchedOutputBytes(0), m_totalOutputBytes(0) {
    resetEvaluator();
}

void TopBitEvaluator::evaluateCircuit(unsigned char* circuitOutputs, unsigned char* referenceOutputs) {
    for (int outputByte = 0; outputByte < pGlobals->settings->circuit.sizeOutputLayer; outputByte++) {
        if (referenceOutputs[0] >> 7 == circuitOutputs[outputByte] >> 7) {
            m_matchedOutputBytes++;
        }
        m_totalOutputBytes++;
    }
}

double TopBitEvaluator::getFitness() {
    return m_matchedOutputBytes / (double) m_totalOutputBytes;
}

void TopBitEvaluator::resetEvaluator() {
    m_matchedOutputBytes = 0;
    m_totalOutputBytes = 0;
}

string TopBitEvaluator::shortDescription() {
    return "top-bit in each output byte evaluator";
}
