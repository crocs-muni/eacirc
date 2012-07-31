#include "../SSGlobals.h"
#include "ICircuitEvaluator.h"
#include "PredictBitCircuitEvaluator.h"
#include "PredictByteCircuitEvaluator.h"
#include "PredictBitGroupParityCircuitEvaluator.h"
#include "PredictBytesParityCircuitEvaluator.h"
#include "PredictHammingWeightCircuitEvaluator.h"
#include "DistinguishTwoEvaluator.h"
#include "PredictAvalancheEvaluator.h"
#include "../EACirc.h"

ICircuitEvaluator::ICircuitEvaluator() {
}

ICircuitEvaluator* ICircuitEvaluator::getCircEvalClass(void) {
	switch (pGACirc->predictMethod) {
		case PREDICT_BIT:
			return new PredictBitCircuitEvaluator();
			break;
		case PREDICT_BITGROUP_PARITY: 
			return new PredictBitGroupParityCircuitEvaluator();
			break;
		case PREDICT_BYTES_PARITY:
			return new PredictBytesParityCircuitEvaluator();
			break;
		case PREDICT_HAMMING_WEIGHT:
			return new PredictHammingWeightCircuitEvaluator();
			break;
		case PREDICT_BYTE:
			return new PredictByteCircuitEvaluator();
			break;
		case PREDICT_DISTINGUISH:
			return new DistinguishTwoEvaluator();
			break;
		case PREDICT_AVALANCHE:
			return new AvalancheEvaluator();
		default:
            assert(FALSE);
			break;
	}
	return NULL;
}