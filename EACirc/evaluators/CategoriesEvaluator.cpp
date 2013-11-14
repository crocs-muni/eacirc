#include "CategoriesEvaluator.h"
#include "CommonFnc.h"

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
    float chiSquareValue = 0;
    int dof = 0;
    // using two-smaple Chi^2 test (http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm)
    float k1 = 1;
    float k2 = 1;
    for (int category = 0; category < pGlobals->settings->main.evaluatorPrecision; category++) {
        if (m_categoriesStream0[category] + m_categoriesStream1[category] > 0) {
            dof++;
            chiSquareValue += pow(k1*m_categoriesStream1[category]-k2*m_categoriesStream0[category], 2) /
                              (m_categoriesStream0[category] + m_categoriesStream1[category]);
        }
    }
    dof--; // last category is fully determined by others
    return (1.0 - chisqr(dof,chiSquareValue));
}

void CategoriesEvaluator::resetEvaluator() {
    m_totalStream0 = m_totalStream1 = 0;
    memset(m_categoriesStream0, 0, pGlobals->settings->main.evaluatorPrecision * sizeof(int));
    memset(m_categoriesStream1, 0, pGlobals->settings->main.evaluatorPrecision * sizeof(int));
}

string CategoriesEvaluator::shortDescription() const {
    return "modular categories evaluator";
}
