#include "CategoriesEvaluator.h"

CategoriesEvaluator::CategoriesEvaluator()
    : IEvaluator(EVALUATOR_CATEGORIES), m_categoriesStream0(NULL), m_categoriesStream1(NULL),
      m_totalStream0(0), m_totalStream1(0), m_numUnderThreshold(0) {
    m_categoriesStream0 = new int[pGlobals->settings->main.evaluatorPrecision];
    m_categoriesStream1 = new int[pGlobals->settings->main.evaluatorPrecision];
    resetEvaluator();
}

CategoriesEvaluator::~CategoriesEvaluator() {
    delete m_categoriesStream0;
    m_categoriesStream0 = NULL;
    delete m_categoriesStream1;
    m_categoriesStream1 = NULL;
    if (m_numUnderThreshold != 0) {
        mainLogger.out(LOGGER_WARNING) << m_numUnderThreshold << " categories under threshold of " << CATEGORY_THRESHOLD << endl;
    }
}

void CategoriesEvaluator::evaluateCircuit(unsigned char* circuitOutputs, unsigned char* referenceOutputs) {
    // select stream map to update and increase total in corresponding counter
    int* currentStreamMap;
    if (referenceOutputs[0] >> (BITS_IN_UCHAR-1) == 0) {
        currentStreamMap = m_categoriesStream0;
        m_totalStream0++;
    } else {
        currentStreamMap = m_categoriesStream1;
        m_totalStream1++;
    }
    // increase category for each output byte (value modulo number of categories)
    for (int outputByte = 0; outputByte < pGlobals->settings->circuit.sizeOutput; outputByte++) {
        currentStreamMap[circuitOutputs[outputByte] % pGlobals->settings->main.evaluatorPrecision]++;
    }
}

float CategoriesEvaluator::getFitness() const {
    float fitness = 0;
    // compute Pearson's Chi square test
    // chi^2 = sum_{i=1}^{n}{\frac{(Observed_i-Expected_i)^2}{Expected_i}}
    // check for threshold E_i >=5, Q_i >=5
    for (int category = 0; category < pGlobals->settings->main.evaluatorPrecision; category++) {
        float divider = max(m_categoriesStream0[category], 1); // prevent division by zero
        fitness += pow(m_categoriesStream1[category]-m_categoriesStream0[category], 2) / divider;
    }
    return fitness;
}

void CategoriesEvaluator::resetEvaluator() {
    unsigned long tempNumUnderThreshold = 0;
    for (int category = 0; category < pGlobals->settings->main.evaluatorPrecision; category++) {
        if (m_categoriesStream0[category] < CATEGORY_THRESHOLD) tempNumUnderThreshold++;
        if (m_categoriesStream1[category] < CATEGORY_THRESHOLD) tempNumUnderThreshold++;
    }
    if (tempNumUnderThreshold != 0) m_numUnderThreshold += tempNumUnderThreshold;
    m_totalStream0 = m_totalStream1 = 0;
    memset(m_categoriesStream0, 0, pGlobals->settings->main.evaluatorPrecision * sizeof(int));
    memset(m_categoriesStream1, 0, pGlobals->settings->main.evaluatorPrecision * sizeof(int));
}

string CategoriesEvaluator::shortDescription() const {
    return "modular categories evaluator";
}
