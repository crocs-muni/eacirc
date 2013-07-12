#include "TopBitEvaluator.h"

TopBitEvaluator::TopBitEvaluator()
    : IEvaluator(EVALUATOR_TOP_BIT), m_matchedOutputBytes(0), m_totalOutputBytes(0) {
    resetEvaluator();
}

void TopBitEvaluator::evaluateCircuit(unsigned char* circuitOutputs, unsigned char* referenceOutputs) {
    for (int outputByte = 0; outputByte < pGlobals->settings->circuit.sizeOutputLayer; outputByte++) {
        if (referenceOutputs[0] >> (BITS_IN_UCHAR-1) == circuitOutputs[outputByte] >> (BITS_IN_UCHAR-1)) {
            m_matchedOutputBytes++;
        }
        m_totalOutputBytes++;
    }
}

float TopBitEvaluator::getFitness() const {
    return m_matchedOutputBytes / (float) m_totalOutputBytes;
}

void TopBitEvaluator::resetEvaluator() {
    m_matchedOutputBytes = 0;
    m_totalOutputBytes = 0;
}

string TopBitEvaluator::shortDescription() const {
    return "top-bit in each output byte evaluator";
}
