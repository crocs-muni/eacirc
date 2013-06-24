#include "CategoriesEvaluator.h"

CategoriesEvaluator::CategoriesEvaluator()
    : IEvaluator(EVALUATOR_CATEGORIES), m_categoriesStream0(NULL), m_categoriesStream1(NULL) {
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
    int* currentStreamMap = referenceOutputs[0] >> 7 == 0 ? m_categoriesStream0 : m_categoriesStream1;
    // increase category for each output byte (value modulo number of categories)
    for (int outputByte = 0; outputByte < pGlobals->settings->circuit.sizeOutputLayer; outputByte++) {
        currentStreamMap[circuitOutputs[outputByte] % pGlobals->settings->main.evaluatorPrecision]++;
    }
}

double CategoriesEvaluator::getFitness() {
    int categoriesUnderThreshold = 0;
    double fitness = 0;
    // add normalised Euclidean distance for each category
    for (int category = 0; category < pGlobals->settings->main.evaluatorPrecision; category++) {
        fitness += pow(m_categoriesStream0[category] - m_categoriesStream1[category], 2) / (float) m_categoriesStream1[category];
        if (m_categoriesStream0[category] < CATEGORY_WARNING_THRESHOLD) {
            categoriesUnderThreshold++;
        }
        if (m_categoriesStream1[category] < CATEGORY_WARNING_THRESHOLD) {
            categoriesUnderThreshold++;
        }
    }
    // transform fitness from interval <0,inf) to interval <0,1>
    fitness = fitness / (1 + fitness);
    // issue warning when categories below 5 numbers
    if (categoriesUnderThreshold != 0) {
        mainLogger.out(LOGGER_WARNING) << "Evaluator: values in " << categoriesUnderThreshold << "/";
        mainLogger.out() << 2*pGlobals->settings->main.evaluatorPrecision << " categories below threshold (";
        mainLogger.out() << CATEGORY_WARNING_THRESHOLD << ")." << endl;
    }
    return fitness;
}

void CategoriesEvaluator::resetEvaluator() {
    memset(m_categoriesStream0, 0, pGlobals->settings->main.evaluatorPrecision);
    memset(m_categoriesStream1, 0, pGlobals->settings->main.evaluatorPrecision);
}

string CategoriesEvaluator::shortDescription() {
    return "modular categories evaluator";
}
