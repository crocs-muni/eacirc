#include "CategoriesEvaluator.h"
#include "CommonFnc.h"

using namespace std;

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
    // Highest bit in referenceOutputs[0] determines which data was used
    // in the evaluation of the circuit (i.e., from algorithm 1 or algorithm 2).
    if (referenceOutputs[0] >> (BITS_IN_UCHAR-1) == 0) {
        currentStreamMap = m_categoriesStream0;
        m_totalStream0++;
    } else {
        currentStreamMap = m_categoriesStream1;
        m_totalStream1++;
    }
    // increase category for each output byte (value modulo number of categories)
    for (int outputByte = 0; outputByte < pGlobals->settings->main.circuitSizeOutput; outputByte++) {
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
        if (m_categoriesStream0[category] + m_categoriesStream1[category] > 5) {
            dof++;
            chiSquareValue += pow(k1*m_categoriesStream1[category]-k2*m_categoriesStream0[category], 2) /
                              (m_categoriesStream0[category] + m_categoriesStream1[category]);
        }
    }
    dof--; // last category is fully determined by others
    float fitness = (1.0 - CommonFnc::chisqr(dof,chiSquareValue));
/*
    // write histogram data in necessary
    if (pGlobals->settings->outputs.verbosity >= LOGGER_VERBOSITY_DEBUG) {
        ofstream hist(FILE_HISTOGRAMS, ios_base::app);
        hist << "Current generation: " << pGlobals->stats.actGener << endl;
        hist << "Stream 0 data: ";
        for (int category = 0; category < pGlobals->settings->main.evaluatorPrecision; category++) {
            hist << m_categoriesStream0[category] << " ";
        }
        hist << endl;
        hist << "Stream 1 data: ";
        for (int category = 0; category < pGlobals->settings->main.evaluatorPrecision; category++) {
            hist << m_categoriesStream1[category] << " ";
        }
        hist << endl;
        hist << "DoF: " << dof << endl;
        hist << "Chi^2: " << (1.0 - fitness) << endl;
        hist << "Fitness: " << fitness << endl;
        hist << endl;
        hist.close();
    }
*/
    return fitness;
}

void CategoriesEvaluator::resetEvaluator() {
    m_totalStream0 = m_totalStream1 = 0;
    memset(m_categoriesStream0, 0, pGlobals->settings->main.evaluatorPrecision * sizeof(int));
    memset(m_categoriesStream1, 0, pGlobals->settings->main.evaluatorPrecision * sizeof(int));
}

string CategoriesEvaluator::shortDescription() const {
    return "modular categories evaluator";
}
