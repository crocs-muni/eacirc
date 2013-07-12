#include "IEvaluator.h"
#include "TopBitEvaluator.h"
#include "CategoriesEvaluator.h"
#include "HammingWeightEvaluator.h"

IEvaluator::IEvaluator(int type) : m_type(type) { }

IEvaluator::~IEvaluator() {}

int IEvaluator::getEvaluatorType() const {
    return m_type;
}

IEvaluator *IEvaluator::getStandardEvaluator(int type) {
    if (type >= EVALUATOR_PROJECT_SPECIFIC_MINIMUM) {
        mainLogger.out(LOGGER_WARNING) << "Evaluator constant in the project range (" << type << ")." << endl;
        return NULL;
    }
    switch (type) {
    case EVALUATOR_HAMMING_WEIGHT:
        return new HammingWeightEvaluator();
        break;
    case EVALUATOR_TOP_BIT:
        return new TopBitEvaluator();
        break;
    case EVALUATOR_CATEGORIES:
        return new CategoriesEvaluator();
        break;
    default:
        mainLogger.out(LOGGER_ERROR) << "Unknown evaluator type \"" << type << "\"." << endl;
        return NULL;
        break;
    }
}
