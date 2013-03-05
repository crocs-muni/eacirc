#include "IEvaluator.h"
#include "PredictBitCircuitEvaluator.h"
#include "PredictByteCircuitEvaluator.h"
#include "PredictBitGroupParityCircuitEvaluator.h"
#include "PredictBytesParityCircuitEvaluator.h"
#include "PredictHammingWeightCircuitEvaluator.h"
#include "DistinguishTwoEvaluator.h"
//#include "PredictAvalancheEvaluator.h"

IEvaluator::IEvaluator(int type) : m_type(type) { }

IEvaluator* IEvaluator::getEvaluator(int type) {
    switch (type) {
        case EVALUATOR_BIT:
			return new PredictBitCircuitEvaluator();
			break;
        case EVALUATOR_BITGROUP_PARITY:
			return new PredictBitGroupParityCircuitEvaluator();
			break;
        case EVALUATOR_BYTES_PARITY:
			return new PredictBytesParityCircuitEvaluator();
			break;
        case EVALUATOR_HAMMING_WEIGHT:
            return new PredictHammingWeightCircuitEvaluator();
			break;
        case EVALUATOR_BYTE:
			return new PredictByteCircuitEvaluator();
			break;
        case EVALUATOR_DISTINGUISH:
			return new DistinguishTwoEvaluator();
			break;
        case EVALUATOR_AVALANCHE:
            return NULL;
            //return new AvalancheEvaluator();
            break;
		default:
            mainLogger.out(LOGGER_ERROR) << "Unknown evaluator type \"" << type << "\"." << endl;
			break;
	}
	return NULL;
}
