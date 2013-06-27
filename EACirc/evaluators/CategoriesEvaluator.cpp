#include "CategoriesEvaluator.h"

CategoriesEvaluator::CategoriesEvaluator()
    : IEvaluator(EVALUATOR_CATEGORIES), m_categoriesStream0(NULL), m_categoriesStream1(NULL),
      m_totalStream0(0), m_totalStream1(0) {
    m_categoriesStream0 = new int[pGlobals->settings->main.evaluatorPrecision];
    m_categoriesStream1 = new int[pGlobals->settings->main.evaluatorPrecision];
    resetEvaluator();
}

CategoriesEvaluator::~CategoriesEvaluator() {
    delete m_categoriesStream0;
    m_categoriesStream0 = NULL;
    delete m_categoriesStream1;
    m_categoriesStream1 = NULL;
}

void CategoriesEvaluator::evaluateCircuit(unsigned char* circuitOutputs, unsigned char* referenceOutputs) {
    // select stream map to update
    int* currentStreamMap = referenceOutputs[0] >> (BITS_IN_UCHAR-1) == 0 ? m_categoriesStream0 : m_categoriesStream1;
    // increase category for each output byte (value modulo number of categories)
    for (int outputByte = 0; outputByte < pGlobals->settings->circuit.sizeOutputLayer; outputByte++) {
        currentStreamMap[circuitOutputs[outputByte] % pGlobals->settings->main.evaluatorPrecision]++;
    }
}

double CategoriesEvaluator::getFitness() {
    double fitness = 0;
    double temp0, temp1;
    // add normalised Euclidean distance for each category
    for (int category = 0; category < pGlobals->settings->main.evaluatorPrecision; category++) {
        temp0 = m_totalStream0 ? m_categoriesStream0[category] / (double) m_totalStream0 : 0;
        temp1 = m_totalStream1 ? m_categoriesStream1[category] / (double) m_totalStream1 : 0;
        fitness += pow(temp0 - temp1, 2);
    }
    // transform fitness from interval <0,2> to interval <0,1>
    fitness = fitness / 2;
    return fitness;
}

void CategoriesEvaluator::resetEvaluator() {
    m_totalStream0 = m_totalStream1 = 0;
    memset(m_categoriesStream0, 0, pGlobals->settings->main.evaluatorPrecision * sizeof(int));
    memset(m_categoriesStream1, 0, pGlobals->settings->main.evaluatorPrecision * sizeof(int));
}

string CategoriesEvaluator::shortDescription() {
    return "modular categories evaluator";
}
