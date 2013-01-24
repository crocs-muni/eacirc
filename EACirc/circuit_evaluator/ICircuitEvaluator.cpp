#include "ICircuitEvaluator.h"
#include "PredictBitCircuitEvaluator.h"
#include "PredictByteCircuitEvaluator.h"
#include "PredictBitGroupParityCircuitEvaluator.h"
#include "PredictBytesParityCircuitEvaluator.h"
#include "PredictHammingWeightCircuitEvaluator.h"
#include "DistinguishTwoEvaluator.h"
#include "PredictAvalancheEvaluator.h"

ICircuitEvaluator::ICircuitEvaluator() {
}

ICircuitEvaluator* ICircuitEvaluator::getCircEvalClass(void) {
    switch (pGACirc->settings->main.evaluatorType) {
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
			return new AvalancheEvaluator();
		default:
            assert(FALSE);
			break;
	}
	return NULL;
}
