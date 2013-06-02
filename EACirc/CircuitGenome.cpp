#include <iomanip>
#include "CircuitGenome.h"
#include "CommonFnc.h"
#include "evaluators/IEvaluator.h"
#include "XMLProcessor.h"
// libinclude ("galib/GAPopulation.h")
#include "GAPopulation.h"
#include "generators/IRndGen.h"

int headerCircuit_inputLayerSize = 19;
int headerCircuit_outputLayerSize = 5;

static void circuit(unsigned char inputs[64], unsigned char outputs[2]) {
    const int SECTOR_SIZE = 16;
    const int NUM_SECTORS = 4;

    unsigned char VAR_IN_0 = 0;
    unsigned char VAR_IN_1 = 0;
    unsigned char VAR_IN_2 = 0;
    unsigned char VAR_IN_3 = 0;
    unsigned char VAR_IN_4 = 0;
    unsigned char VAR_IN_5 = 0;
    unsigned char VAR_IN_6 = 0;
    unsigned char VAR_IN_7 = 0;
    unsigned char VAR_IN_8 = 0;
    unsigned char VAR_IN_9 = 0;
    unsigned char VAR_IN_10 = 0;
    unsigned char VAR_IN_11 = 0;
    unsigned char VAR_IN_12 = 0;
    unsigned char VAR_IN_13 = 0;
    unsigned char VAR_IN_14 = 0;
    unsigned char VAR_IN_15 = 0;
    unsigned char VAR_IN_16 = 0;
    unsigned char VAR_IN_17 = 0;
    unsigned char VAR_IN_18 = 0;

    for (int sector = 0; sector < NUM_SECTORS; sector++) {
        VAR_IN_3 = inputs[sector * SECTOR_SIZE + 0];
        VAR_IN_4 = inputs[sector * SECTOR_SIZE + 1];
        VAR_IN_5 = inputs[sector * SECTOR_SIZE + 2];
        VAR_IN_6 = inputs[sector * SECTOR_SIZE + 3];
        VAR_IN_7 = inputs[sector * SECTOR_SIZE + 4];
        VAR_IN_8 = inputs[sector * SECTOR_SIZE + 5];
        VAR_IN_9 = inputs[sector * SECTOR_SIZE + 6];
        VAR_IN_10 = inputs[sector * SECTOR_SIZE + 7];
        VAR_IN_11 = inputs[sector * SECTOR_SIZE + 8];
        VAR_IN_12 = inputs[sector * SECTOR_SIZE + 9];
        VAR_IN_13 = inputs[sector * SECTOR_SIZE + 10];
        VAR_IN_14 = inputs[sector * SECTOR_SIZE + 11];
        VAR_IN_15 = inputs[sector * SECTOR_SIZE + 12];
        VAR_IN_16 = inputs[sector * SECTOR_SIZE + 13];
        VAR_IN_17 = inputs[sector * SECTOR_SIZE + 14];
        VAR_IN_18 = inputs[sector * SECTOR_SIZE + 15];

        unsigned char VAR_1_0_SUM = VAR_IN_0 + 0;
        unsigned char VAR_1_1_XOR = VAR_IN_1 ^ VAR_IN_17 ^ 0;
        unsigned char VAR_1_2_NOR = 0 | ~ VAR_IN_2 | ~ VAR_IN_5 | ~ 0xff;
        unsigned char VAR_1_3_ROR_7 = VAR_IN_1 >> 7 ;
        unsigned char VAR_1_4_ROL_5 = VAR_IN_0 << 5 ;
        unsigned char VAR_1_5_XOR = VAR_IN_5 ^ 0;
        unsigned char VAR_1_6_NOP = VAR_IN_6  ;
        unsigned char VAR_1_7_OR_ = VAR_IN_7 | 0;
        unsigned char VAR_1_8_NOR = 0 | ~ 0xff;
        unsigned char VAR_1_9_XOR = VAR_IN_0 ^ VAR_IN_4 ^ VAR_IN_9 ^ VAR_IN_18 ^ 0;
        unsigned char VAR_1_10_ADD = VAR_IN_9 + VAR_IN_10 + 0;
        unsigned char VAR_1_11_SUB = VAR_IN_11 - VAR_IN_15 - VAR_IN_18 - 0;
        unsigned char VAR_1_12_BSL_7 = VAR_IN_3  & 223 ;
        unsigned char VAR_1_13_NAN = 0xff & ~  VAR_IN_1 & ~ VAR_IN_7 & ~ VAR_IN_13 & ~ 0;
        unsigned char VAR_1_14_DIV = ((VAR_IN_14 != 0) ? VAR_IN_14 : 1) / ((VAR_IN_15 != 0) ? VAR_IN_15 : 1) / 1;
        unsigned char VAR_1_15_SUB = VAR_IN_12 - VAR_IN_15 - 0;
        unsigned char VAR_1_16_ADD = VAR_IN_8 + VAR_IN_15 + VAR_IN_16 + 0;
        unsigned char VAR_1_17_ROL_5 = VAR_IN_17 << 5 ;
        unsigned char VAR_1_18_SUM = VAR_IN_13 + VAR_IN_18 + 0;
        unsigned char VAR_1_19_NOR = 0 | ~ VAR_IN_5 | ~ VAR_IN_13 | ~ 0xff;
        unsigned char VAR_2_0_NAN = 0xff & ~  VAR_1_0_SUM & ~ VAR_1_1_XOR & ~ 0;
        unsigned char VAR_2_1_OR_ = VAR_1_0_SUM | VAR_1_1_XOR | VAR_1_2_NOR | VAR_1_3_ROR_7 | 0;
        unsigned char VAR_2_2_NOR = 0 | ~ 0xff;
        unsigned char VAR_2_3_SUM = VAR_1_2_NOR + VAR_1_3_ROR_7 + VAR_1_5_XOR + 0;
        unsigned char VAR_2_4_ROL_7 = 0;
        unsigned char VAR_2_5_NOP = VAR_1_6_NOP  ;
        unsigned char VAR_2_6_MUL = VAR_1_6_NOP * VAR_1_7_OR_ * VAR_1_8_NOR * 1;
        unsigned char VAR_2_7_ADD = VAR_1_7_OR_ + VAR_1_8_NOR + 0;
        unsigned char VAR_2_8_NOR = 0 | ~ 0xff;
        unsigned char VAR_2_9_NAN = 0xff & ~  0;
        unsigned char VAR_2_10_NOP = VAR_1_9_XOR  ;
        unsigned char VAR_2_11_OR_ = VAR_1_10_ADD | VAR_1_11_SUB | VAR_1_12_BSL_7 | VAR_1_13_NAN | 0;
        unsigned char VAR_2_12_ROL_6 = VAR_1_11_SUB << 6 ;
        unsigned char VAR_2_13_XOR = VAR_1_13_NAN ^ 0;
        unsigned char VAR_2_14_SUM = VAR_1_15_SUB + 0;
        unsigned char VAR_2_15_DIV = 0;
        unsigned char VAR_2_16_ADD = VAR_1_16_ADD + VAR_1_17_ROL_5 + 0;
        unsigned char VAR_2_17_ROR_3 = VAR_1_18_SUM >> 3 ;
        unsigned char VAR_2_18_NOR = 0 | ~ VAR_1_17_ROL_5 | ~ VAR_1_18_SUM | ~ VAR_1_19_NOR | ~ 0xff;
        unsigned char VAR_2_19_NOR = 0 | ~ VAR_1_17_ROL_5 | ~ VAR_1_19_NOR | ~ 0xff;
        unsigned char VAR_3_0_SUM = VAR_2_2_NOR + 0;
        unsigned char VAR_3_1_BSL_0 = VAR_2_0_NAN  & 0 ;
        unsigned char VAR_3_2_NOR = 0 | ~ VAR_2_2_NOR | ~ VAR_2_3_SUM | ~ 0xff;
        unsigned char VAR_3_3_XOR = VAR_2_3_SUM ^ 0;
        unsigned char VAR_3_4_ROL_0 = VAR_2_5_NOP << 0 ;
        unsigned char VAR_3_5_NAN = 0xff & ~  VAR_2_4_ROL_7 & ~ VAR_2_5_NOP & ~ VAR_2_7_ADD & ~ 0;
        unsigned char VAR_3_6_MUL = VAR_2_6_MUL * 1;
        unsigned char VAR_3_7_ROR_7 = VAR_2_9_NAN >> 7 ;
        unsigned char VAR_3_8_SUB = 0;
        unsigned char VAR_3_9_XOR = VAR_2_8_NOR ^ VAR_2_11_OR_ ^ 0;
        unsigned char VAR_3_10_DIV = ((VAR_2_11_OR_ != 0) ? VAR_2_11_OR_ : 1) / 1;
        unsigned char VAR_3_11_ROR_0 = VAR_2_10_NOP >> 0 ;
        unsigned char VAR_3_12_XOR = VAR_2_11_OR_ ^ 0;
        unsigned char VAR_3_13_ROL_3 = VAR_2_15_DIV << 3 ;
        unsigned char VAR_3_14_CONST_191 = 191 ;
        unsigned char VAR_3_15_DIV = ((VAR_2_14_SUM != 0) ? VAR_2_14_SUM : 1) / ((VAR_2_16_ADD != 0) ? VAR_2_16_ADD : 1) / ((VAR_2_17_ROR_3 != 0) ? VAR_2_17_ROR_3 : 1) / 1;
        unsigned char VAR_3_16_AND = VAR_2_17_ROR_3 & 0xff;
        unsigned char VAR_3_17_NOR = 0 | ~ VAR_2_16_ADD | ~ VAR_2_18_NOR | ~ VAR_2_19_NOR | ~ 0xff;
        unsigned char VAR_3_18_SUB = VAR_2_16_ADD - VAR_2_18_NOR - VAR_2_19_NOR - 0;
        unsigned char VAR_3_19_ROR_0 = VAR_2_18_NOR >> 0 ;
        unsigned char VAR_4_0_OR_ = VAR_3_1_BSL_0 | VAR_3_3_XOR | 0;
        unsigned char VAR_4_1_NAN = 0xff & ~  VAR_3_1_BSL_0 & ~ VAR_3_3_XOR & ~ 0;
        unsigned char VAR_4_2_SUB = VAR_3_1_BSL_0 - VAR_3_3_XOR - 0;
        unsigned char VAR_4_3_ADD = VAR_3_4_ROL_0 + VAR_3_5_NAN + 0;
        unsigned char VAR_4_4_OR_ = VAR_3_3_XOR | VAR_3_4_ROL_0 | VAR_3_5_NAN | 0;
        unsigned char VAR_4_5_NOR = 0 | ~ VAR_3_4_ROL_0 | ~ VAR_3_6_MUL | ~ VAR_3_7_ROR_7 | ~ 0xff;
        unsigned char VAR_4_6_CONST_127 = 127 ;
        unsigned char VAR_4_7_NOR = 0 | ~ VAR_3_8_SUB | ~ 0xff;
        unsigned char VAR_4_8_XOR = VAR_3_7_ROR_7 ^ VAR_3_9_XOR ^ VAR_3_10_DIV ^ 0;
        unsigned char VAR_4_9_SUB = VAR_3_8_SUB - VAR_3_9_XOR - VAR_3_10_DIV - 0;
        unsigned char VAR_4_10_CONST_255 = 255 ;
        unsigned char VAR_4_11_SUM = VAR_3_10_DIV + VAR_3_11_ROR_0 + VAR_3_13_ROL_3 + 0;
        unsigned char VAR_4_12_SUM = VAR_3_11_ROR_0 + VAR_3_12_XOR + VAR_3_13_ROL_3 + VAR_3_14_CONST_191 + 0;
        unsigned char VAR_4_13_OR_ = VAR_3_12_XOR | VAR_3_15_DIV | 0;
        unsigned char VAR_4_14_ROL_7 = VAR_3_15_DIV << 7 ;
        unsigned char VAR_4_15_BSL_7 = VAR_3_15_DIV  & 183 ;
        unsigned char VAR_4_16_BSL_0 = VAR_3_16_AND  & 192 ;
        unsigned char VAR_4_17_OR_ = 0;
        unsigned char VAR_4_18_ADD = VAR_3_18_SUB + 0;
        unsigned char VAR_4_19_ADD = VAR_3_16_AND + 0;
        unsigned char VAR_5_0_AND = VAR_4_1_NAN & VAR_4_2_SUB & 0xff;
        unsigned char VAR_5_1_AND = VAR_4_4_OR_ & 0xff;
        unsigned char VAR_5_2_SUM = VAR_4_1_NAN + VAR_4_2_SUB + VAR_4_4_OR_ + 0;
        unsigned char VAR_5_3_NAN = 0xff & ~  VAR_4_0_OR_ & ~ VAR_4_16_BSL_0 & ~ 0;
        unsigned char VAR_5_4_ADD = VAR_4_1_NAN + VAR_4_7_NOR + 0;

        VAR_IN_0 = VAR_5_0_AND;
        VAR_IN_1 = VAR_5_1_AND;
        VAR_IN_2 = VAR_5_2_SUM;
        outputs[0] = VAR_5_3_NAN;
        outputs[1] = VAR_5_4_ADD;
    }

}

float CircuitGenome::Evaluator(GAGenome &g) {
    int								status = STAT_OK;
    GA1DArrayGenome<GENOM_ITEM_TYPE>  &genome = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) g;
    float							fitness = 0;
    unsigned char*					usePredictorMask = NULL;
    int								match = 0;
    int								numPredictions = 0; 
    IEvaluator*                     evaluator = pGlobals->evaluator;

	/* DO NOT USE PREDICTOR MASK
	int memoryLength = (pGlobals->settings->circuit.useMemory) ? pGlobals->settings->circuit.memorySize : 0;
    usePredictorMask = new unsigned char[pGlobals->settings->circuit.totalSizeOutputLayer + memoryLength];
    memset(usePredictorMask, 1, pGlobals->settings->circuit.totalSizeOutputLayer + memoryLength);	// USE ALL PREDICTORS
	*/
    match = 0;    
    numPredictions = 0;
    for (int testVector = 0; testVector < pGlobals->settings->testVectors.setSize; testVector++) {            
        // EXECUTE CIRCUIT
        status = ExecuteCircuit(&genome, pGlobals->testVectors.inputs[testVector], pGlobals->testVectors.circuitOutputs[testVector]);

        // EVALUATE SUCCESS OF CIRCUIT FOR THIS TEST VECTOR
        evaluator->evaluateCircuit(pGlobals->testVectors.circuitOutputs[testVector], pGlobals->testVectors.outputs[testVector],
                                   usePredictorMask, &match, &numPredictions);
    }
    fitness = (numPredictions > 0) ? (match / ((float) numPredictions)) : 0;

    // update statistics, if needed
    if (!pGlobals->stats.prunningInProgress) {
        // include into average fitness of whole generation
        (pGlobals->stats.avgGenerFit) += fitness;
        (pGlobals->stats.numAvgGenerFit)++;
        (pGlobals->stats.avgPredictions) += numPredictions;

        if (fitness > pGlobals->stats.bestGenerFit) pGlobals->stats.bestGenerFit = fitness;

        if (pGlobals->stats.maxFit < fitness) {
            pGlobals->stats.maxFit = fitness;

            // DISPLAY CURRENTLY BEST
            ostringstream os2;
            os2 << FILE_CIRCUIT << setprecision(CIRCUIT_FILENAME_PRECISION) << fixed << fitness;
            string filePath = os2.str();
            PrintCircuit(genome, filePath, usePredictorMask, FALSE);   // PRINT WITHOUT PRUNNING

            if (pGlobals->settings->circuit.allowPrunning) {
                filePath += "_prunned";
                PrintCircuit(genome, filePath, usePredictorMask, TRUE);    // PRINT WITH PRUNNING
            }
        }
    }
    delete[] usePredictorMask;
    return fitness;
}

void CircuitGenome::Initializer(GAGenome& g) {
	CircuitGenome::Initializer_basic(g);
}

void CircuitGenome::Initializer_basic(GAGenome& g) {
    GA1DArrayGenome<GENOM_ITEM_TYPE>& genome = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) g;
    int	offset = 0;

	// CLEAR GENOM
	for (int i = 0; i < genome.size(); i++) genome.gene(i, 0);
    
	// INITIALIZE ALL LAYERS
    for (int layer = 0; layer < 2 * pGlobals->settings->circuit.numLayers; layer++) {
        offset = layer * pGlobals->settings->circuit.sizeLayer;
        int numLayerInputs = 0;
        if (layer == 0) numLayerInputs = pGlobals->settings->circuit.sizeInputLayer;
        else numLayerInputs = pGlobals->settings->circuit.sizeLayer;
        
        int numFncInLayer = ((layer == 2 * pGlobals->settings->circuit.numLayers - 1) || (layer == 2 * pGlobals->settings->circuit.numLayers - 2)) ? pGlobals->settings->circuit.totalSizeOutputLayer : pGlobals->settings->circuit.sizeLayer;

        for (int slot = 0; slot < numFncInLayer; slot++) {
            GENOM_ITEM_TYPE   value;
            if (layer % 2 == 0) {
                // CONNECTION SUB-LAYER
                if (layer / 2 == 0) {
                    // IN_SELECTOR_LAYER - TAKE INPUT ONLY FROM PREVIOUS NODE IN SAME COORDINATES
					// NOTE: relative mask must be computed (in relative masks, (numLayerInputs / 2) % numLayerInputs is node at same column)
					int relativeSlot = (numLayerInputs / 2) % numLayerInputs;
                    value = pGlobals->precompPow[relativeSlot];
                }
                else {
                    // FUNCTIONAL LAYERS (CONNECTOR_LAYER_i)
					galibGenerator->getRandomFromInterval(pGlobals->precompPow[pGlobals->settings->circuit.numConnectors], &value);
                }
                genome.gene(offset + slot, value);
            }
            else {
                // FUNCTION SUB-LAYER, SET ONLY ALLOWED FUNCTIONS  
                
				if (layer / 2 == 0) {
                    // FUNCTION_LAYER_1 - PASS INPUT WITHOUT CHANGES (ONLY SIMPLE COMPOSITION OF INPUTS IS ALLOWED)
                    genome.gene(offset + slot, FNC_XOR);
                }
                else {
                    // FUNCTION_LAYER_i
                    int bFNCNotSet = TRUE;
                    while (bFNCNotSet) {
                        galibGenerator->getRandomFromInterval(FNC_MAX, &value);
                        if (pGlobals->settings->circuit.allowedFunctions[value] != 0) {
                            genome.gene(offset + slot, value);
                            bFNCNotSet = FALSE;
                        }
                    }
					// BUGBUG: ARGUMENT1 is always 0
                }
            }
        }
    }
}

int CircuitGenome::Mutator(GAGenome &g, float pmut) {
    GA1DArrayGenome<GENOM_ITEM_TYPE> &genome = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) g;
    int result = 0;
    
    for (int layer = 0; layer < 2 * pGlobals->settings->circuit.numLayers; layer++) {
        int offset = layer * pGlobals->settings->circuit.sizeLayer;

        int numLayerInputs = 0;
        if (layer == 0) numLayerInputs = pGlobals->settings->circuit.sizeInputLayer;
        else numLayerInputs = pGlobals->settings->circuit.sizeLayer;

		// Common layers have SETTINGS_CIRCUIT::sizeLayer functions, output layer have SETTINGS_CIRCUIT::totalSizeOutputLayer
        int numFncInLayer = ((layer == 2 * pGlobals->settings->circuit.numLayers - 1) || (layer == 2 * pGlobals->settings->circuit.numLayers - 2)) ? pGlobals->settings->circuit.totalSizeOutputLayer : pGlobals->settings->circuit.sizeLayer;

        for (int slot = 0; slot < numFncInLayer; slot++) {
            GENOM_ITEM_TYPE value = 0;
            if (layer % 2 == 0) {
                // CONNECTION SUB-LAYER (CONNECTOR_LAYER_i)
                
                // MUTATE CONNECTION SELECTOR (FLIP ONE SINGLE BIT == ONE CONNECTOR)
				// BUGBUG: we are not taking into account restrictions given by SETTINGS_CIRCUIT::numConnectors
                if (GAFlipCoin(pmut)) {
                    GENOM_ITEM_TYPE temp;
                    galibGenerator->getRandomFromInterval(numLayerInputs, &value);
                    temp = pGlobals->precompPow[value];
                    // SWITCH RANDOMLY GENERATED BIT
                    temp ^= genome.gene(offset + slot);
                    genome.gene(offset + slot, temp);
                }
            }
            else {
                // FUNCTION_LAYER_i - MUTATE FUNCTION TYPE USING ONLY ALLOWED FNCs
                GENOM_ITEM_TYPE origValue = genome.gene(offset + slot);
				// MUTATE FNC
				if (GAFlipCoin(pmut)) {             
                    unsigned char newFncValue = 0;
					int bFNCNotSet = TRUE;
                    while (bFNCNotSet) {
                        galibGenerator->getRandomFromInterval(FNC_MAX, &newFncValue);
                        if (pGlobals->settings->circuit.allowedFunctions[newFncValue] != 0) {
							SET_FNC_TYPE(&origValue, newFncValue);
                            bFNCNotSet = FALSE;
                        }
                    }
                    genome.gene(offset + slot, origValue);
                }
				// MUTATE FNC ARGUMENT1
				if (GAFlipCoin(pmut)) {             
					origValue = genome.gene(offset + slot);

					unsigned char newArg1Value = 0;
                    galibGenerator->getRandomFromInterval(0xff, &newArg1Value);
                    // SWITCH RANDOMLY GENERATED BITS
                    newArg1Value = newArg1Value ^ GET_FNC_ARGUMENT1(genome.gene(offset + slot));
					SET_FNC_ARGUMENT1(&origValue, newArg1Value);
					genome.gene(offset + slot, origValue);
                }
            }
        }
    }

    return result;
}

int CircuitGenome::Crossover(const GAGenome &p1, const GAGenome &p2, GAGenome *o1, GAGenome *o2) {
	return Crossover_perColumn(p1, p2, o1, o2);
}

int CircuitGenome::Crossover_perLayer(const GAGenome &p1, const GAGenome &p2, GAGenome *o1, GAGenome *o2) {
    GA1DArrayGenome<GENOM_ITEM_TYPE> &parent1 = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) p1;
    GA1DArrayGenome<GENOM_ITEM_TYPE> &parent2 = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) p2;
    GA1DArrayGenome<GENOM_ITEM_TYPE> &offspring1 = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) *o1;
    GA1DArrayGenome<GENOM_ITEM_TYPE> &offspring2 = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) *o2;
    
    // CROSS ONLY WHOLE LAYERS
    int cpoint = GARandomInt(1,pGlobals->settings->circuit.numLayers) * 2; // point of crossover (from, to)
    for (int layer = 0; layer < 2 * pGlobals->settings->circuit.numLayers; layer++) {
        int offset = layer * pGlobals->settings->circuit.sizeLayer;

        if (layer <= cpoint) {
            for (int i = 0; i < pGlobals->settings->circuit.sizeLayer; i++) {
                if (o1 != NULL) offspring1.gene(offset + i, parent1.gene(offset + i));
                if (o2 != NULL) offspring2.gene(offset + i, parent2.gene(offset + i));
            }
        }
        else {
            for (int i = 0; i < pGlobals->settings->circuit.sizeLayer; i++) {
                if (o1 != NULL) offspring1.gene(offset + i, parent2.gene(offset + i));
                if (o2 != NULL) offspring2.gene(offset + i, parent1.gene(offset + i));
            }
        }
    }
    return 1;
}
int CircuitGenome::Crossover_perColumn(const GAGenome &p1, const GAGenome &p2, GAGenome *o1, GAGenome *o2) {
    GA1DArrayGenome<GENOM_ITEM_TYPE> &parent1 = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) p1;
    GA1DArrayGenome<GENOM_ITEM_TYPE> &parent2 = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) p2;
    GA1DArrayGenome<GENOM_ITEM_TYPE> &offspring1 = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) *o1;
    GA1DArrayGenome<GENOM_ITEM_TYPE> &offspring2 = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) *o2;
    
    // CROSS 
    int cpoint = GARandomInt(1,pGlobals->settings->circuit.sizeLayer); 
    for (int layer = 0; layer < 2 * pGlobals->settings->circuit.numLayers; layer++) {
        int offset = layer * pGlobals->settings->circuit.sizeLayer;
        for (int i = 0; i < pGlobals->settings->circuit.sizeLayer; i++) {
			if ( i < cpoint) {
				if (o1 != NULL) offspring1.gene(offset + i, parent1.gene(offset + i));
				if (o2 != NULL) offspring2.gene(offset + i, parent2.gene(offset + i));
			}
			else {
                if (o1 != NULL) offspring1.gene(offset + i, parent2.gene(offset + i));
                if (o2 != NULL) offspring2.gene(offset + i, parent1.gene(offset + i));
			}
        }
    }
    return 1;
}


int CircuitGenome::GetFunctionLabel(GENOM_ITEM_TYPE functionID, GENOM_ITEM_TYPE connections, string* pLabel) {
    int		status = STAT_OK;
    switch (GET_FNC_TYPE(functionID)) {
        case FNC_NOP: *pLabel = "NOP"; break;
        case FNC_OR: *pLabel = "OR_"; break;
        case FNC_AND: *pLabel = "AND"; break;
        case FNC_CONST: {
			std::stringstream out;
			out << (GET_FNC_ARGUMENT1(functionID)  & 0xff);
			*pLabel = "CONST_" + out.str();
            break;
        }
		case FNC_READX: *pLabel = "RDX"; break;
        case FNC_XOR: *pLabel = "XOR"; break;
        case FNC_NOR: *pLabel = "NOR"; break;
        case FNC_NAND: *pLabel = "NAN"; break;
        case FNC_ROTL: {
			std::stringstream out;
			unsigned char tmp = GET_FNC_ARGUMENT1(functionID);
			out << (GET_FNC_ARGUMENT1(functionID) & 0x07);
            *pLabel = "ROL_" + out.str(); 
            break;
        }
        case FNC_ROTR: {
			std::stringstream out;
			out << (GET_FNC_ARGUMENT1(functionID) & 0x07);
            *pLabel = "ROR_" + out.str(); 
            break;
        }
        case FNC_BITSELECTOR: {
			std::stringstream out;
			out << (GET_FNC_ARGUMENT1(functionID) & 0x07);
            *pLabel = "BSL_" + out.str(); 
            break;
        }
        case FNC_SUM: *pLabel = "SUM"; break;
        case FNC_SUBS: *pLabel = "SUB"; break;
        case FNC_ADD: *pLabel = "ADD"; break;
        case FNC_MULT: *pLabel = "MUL"; break;
        case FNC_DIV: *pLabel = "DIV"; break;
        case FNC_EQUAL: *pLabel = "EQUAL"; break;
        default: {
            assert(FALSE);
            *pLabel = "ERR";
            status = STAT_USERDATA_BAD;
        }
    }
    
    return status;
}

int CircuitGenome::PruneCircuit(GAGenome &g, GAGenome &prunnedG) {
	return PruneCircuitNew(g, prunnedG);
    int                     status = STAT_OK;
    GA1DArrayGenome<GENOM_ITEM_TYPE>  &genome = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) g;
    GA1DArrayGenome<GENOM_ITEM_TYPE>  &prunnedGenome = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) prunnedG;
    int                    bChangeDetected = FALSE;

    // CREATE LOCAL COPY
    for (int i = 0; i < genome.size(); i++) {
        prunnedGenome.gene(i, genome.gene(i)); 
    }

    if (pGlobals->stats.prunningInProgress) {
        // WE ARE ALREADY PERFORMING PRUNING - DO NOT CONTINUE TO PREVENT OVERLAPS
    }
    else {
        //
        // METHOD - TRY TO TEMPORARY REMOVE CONNECTION/FUNCTION AND TEST FITNESS CHANGES
        //
        
        pGlobals->stats.prunningInProgress = true;
        
		float origFit = Evaluator(prunnedGenome);
        
        int prunneRepeat = 0; 
        bChangeDetected = TRUE;
        while (bChangeDetected && prunneRepeat < 10) {
            bChangeDetected = FALSE;
            prunneRepeat++;
            
            // DISABLE GENES STARTING FROM END 
            for (int i = prunnedGenome.size() - 1; i >= 0; i--) {
                GENOM_ITEM_TYPE   origValue = prunnedGenome.gene(i);
                
                if (origValue != 0) {
                    // PRUNE FNC AND CONNECTION LAYER DIFFERENTLY
                    if (((i / pGlobals->settings->circuit.sizeLayer) % 2) == 1) {
                        // FNCs LAYER - TRY TO SET AS NOP INSTRUCTION WITH NO CONNECTORS
                        prunnedGenome.gene(i, FNC_NOP);
                        
                        assert(GET_FNC_TYPE(origValue) <= FNC_MAX);
                        
                        float newFit = Evaluator(prunnedGenome);
                        if (origFit > newFit) {
                            // GENE WAS IMPORTANT, SET BACK 
                            prunnedGenome.gene(i, origValue);
                        }
                        else {
                            bChangeDetected = TRUE;
                            if (origFit < newFit) {
                                // GENE WAS HARMING, KEEP REMOVED
                            }
                            else {
                                // GENE WAS NOT IMPORTANT, KEEP REMOVED
                            }
                        }
                    }
                    else {
                        GENOM_ITEM_TYPE   tempOrigValue = origValue;  // WILL HOLD MASK OF IMPORTANT CONNECTIONS
                        // CONNECTION LAYER - TRY TO REMOVE CONNECTIONS GRADUALLY
                        for (int conn = 0; conn < MAX_LAYER_SIZE; conn++) {
                            GENOM_ITEM_TYPE   newValue = tempOrigValue & (~pGlobals->precompPow[conn]);
                            
                            if (newValue != tempOrigValue) {
                                prunnedGenome.gene(i, newValue);
                                
                                float newFit = Evaluator(prunnedGenome);
                                if (origFit > newFit) {
                                    // GENE WAS IMPORTANT, DO NOT REMOVE CONNECTION
                                }
                                else {
                                    bChangeDetected = TRUE;
                    
                                    // STORE NEW VALUE WITHOUT CONNECTION
                                    tempOrigValue = newValue;
                                    
                                    if (origFit < newFit) {
                                        // GENE WAS HARMING, REMOVED
                                    }
                                    else {
                                        // GENE WAS NOT IMPORTANT, REMOVE
                                    }
                                }
                            }
                        }
                        
                        // SET FINAL PRUNNED VALUE
                        prunnedGenome.gene(i, tempOrigValue);
                    }
                }
            }
        }
    }
    pGlobals->stats.prunningInProgress = false;
    return status;
}

int CircuitGenome::PruneCircuitNew(GAGenome &g, GAGenome &prunnedG) {
    int                     status = STAT_OK;
    GA1DArrayGenome<GENOM_ITEM_TYPE>  &genome = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) g;
    GA1DArrayGenome<GENOM_ITEM_TYPE>  &prunnedGenome = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) prunnedG;
    int                    bChangeDetected = FALSE;

    // CREATE LOCAL COPY
    for (int i = 0; i < genome.size(); i++) {
        prunnedGenome.gene(i, genome.gene(i)); 
    }

    if (pGlobals->stats.prunningInProgress) {
        // WE ARE ALREADY PERFORMING PRUNING - DO NOT CONTINUE TO PREVENT OVERLAPS
    }
    else {
        //
        // METHOD - TRY TO TEMPORARY REMOVE CONNECTION/FUNCTION AND TEST FITNESS CHANGES
        //
        
        pGlobals->stats.prunningInProgress = true;
        
		float origFit = Evaluator(prunnedGenome);
        
        int prunneRepeat = 0; 
        bChangeDetected = TRUE;
        while (bChangeDetected && prunneRepeat < 10) {
            bChangeDetected = FALSE;
            prunneRepeat++;
            
			// DISABLE GENES STARTING FROM END 
			for (int layer = 1; layer < 2 * pGlobals->settings->circuit.numLayers; layer = layer + 2) {
				int offsetCON = (layer-1) * pGlobals->settings->circuit.sizeLayer;
				int offsetFNC = (layer) * pGlobals->settings->circuit.sizeLayer;

				// actual number of functions in layer - different for the last "output" layer
				int numFncInLayer = (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) ? (pGlobals->settings->circuit.totalSizeOutputLayer) : pGlobals->settings->circuit.sizeLayer;

				for (int slot = 0; slot < numFncInLayer; slot++) {
					GENOM_ITEM_TYPE   origValueFnc = genome.gene(offsetFNC + slot);
					GENOM_ITEM_TYPE   origValueCon = genome.gene(offsetCON + slot);

					// TRY TO SET AS NOP INSTRUCTION WITH NO CONNECTORS
					prunnedGenome.gene(offsetFNC + slot, FNC_NOP);	// NOP
					prunnedGenome.gene(offsetCON + slot, 0);		// NO CONNECTORS

                    float newFit = Evaluator(prunnedGenome);
                    if (origFit > newFit) {
                        // SOME PART OF THE GENE WAS IMPORTANT, SET BACK 
						prunnedGenome.gene(offsetFNC + slot, origValueFnc);	
						prunnedGenome.gene(offsetCON + slot, origValueCon);		

                        // TRY TO REMOVE CONNECTIONS GRADUALLY
						GENOM_ITEM_TYPE   prunnedConnectors = origValueCon;
                        for (int conn = 0; conn < MAX_LAYER_SIZE; conn++) {
                            GENOM_ITEM_TYPE   newConValue = prunnedConnectors & (~pGlobals->precompPow[conn]);
                            
                            if (newConValue != prunnedConnectors) {
                                prunnedGenome.gene(offsetCON + slot, newConValue);
                                
                                float newFit = Evaluator(prunnedGenome);
                                if (origFit > newFit) {
                                    // CONNECTOR WAS IMPORTANT, DO NOT REMOVE CONNECTION
                                }
                                else {
                                    bChangeDetected = TRUE;
                    
                                    // STORE NEW VALUE WITHOUT UNIMPORTANT CONNECTION
                                    prunnedConnectors = newConValue;
                                    
                                    if (origFit < newFit) {
                                        // CONNECTOR WAS HARMING, REMOVED
                                    }
                                    else {
                                        // CONNECTOR WAS NOT IMPORTANT, REMOVE
                                    }
                                }
                            }
                        }
                        // SET FINAL PRUNNED VALUE FOR CONNECTORS
                        prunnedGenome.gene(offsetCON + slot, prunnedConnectors);
                    }
					else {
						// GENE WAS UNIMPORTANT, REMOVE IT
					}
                }
            }
        }
    }
    pGlobals->stats.prunningInProgress = false;
    return status;
}

int CircuitGenome::GetUsedNodes(GAGenome &g, unsigned char* usePredictorMask, unsigned char displayNodes[]) {
	int	status = STAT_OK;
    GA1DArrayGenome<GENOM_ITEM_TYPE>  &genome = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) g;
	
	//
	// BUILD SET OF USED NODES FROM OUTPUT TO INPUT
	//
	
	// ADD OUTPUT NODES
    // VISUAL CIRC: CONNECT OUTPUT LAYER
    int offsetFNC = (2 * pGlobals->settings->circuit.numLayers - 1) * pGlobals->settings->circuit.sizeLayer;
    for (int i = 0; i < pGlobals->settings->circuit.totalSizeOutputLayer; i++) {
		if (usePredictorMask == NULL || usePredictorMask[i] == 1) {
			// ADD THIS ONE TO LIST OF USED NODES 
			displayNodes[offsetFNC + i] = 1;	
		}
    }
	
	// PROCESS ALL LAYERS FROM BACK
    for (int layer = 2 * pGlobals->settings->circuit.numLayers - 1; layer > 0; layer = layer - 2) {
        int offsetCON = (layer-1) * pGlobals->settings->circuit.sizeLayer;
        int offsetFNC = (layer) * pGlobals->settings->circuit.sizeLayer;

        // actual number of inputs for this layer. For first layer equal to pCircuit->numInputs, for next layers equal to number of function in intermediate layer pCircuit->internalLayerSize
        int numLayerInputs = 0;
        if (layer == 1) {
            // WHEN SECTORING IS USED THEN FIRST LAYER OBTAIN internalLayerSize INPUTS
            //if (pGACirc->bSectorInputData) numLayerInputs = pGACirc->internalLayerSize; 
            numLayerInputs = pGlobals->settings->circuit.sizeInputLayer;
        }
        else numLayerInputs = pGlobals->settings->circuit.sizeLayer;

        // actual number of functions in layer - different for the last "output" layer
        int numFncInLayer = (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) ? pGlobals->settings->circuit.totalSizeOutputLayer : pGlobals->settings->circuit.sizeLayer;

        // ORDINARY LAYERS HAVE SPECIFIED NUMBER SETTINGS_CIRCUIT::numConnectors
        int	numLayerConnectors = pGlobals->settings->circuit.numConnectors;
		// IN_SELECTOR_LAYER HAS FULL INTERCONNECTION (CONNECTORS)
        if (layer / 2 == 0) numLayerConnectors = numLayerInputs;    
		// OUT_SELECTOR_LAYER HAS FULL INTERCONNECTION (CONNECTORS)
		if (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) numLayerConnectors = numLayerInputs;    
		// IF NUMBER OF CONNECTORS IS HIGHER THAN NUMBER OF FUNCTIONS IN LAYER => LIMIT TO numLayerInputs
		if (numLayerConnectors > numLayerInputs) numLayerConnectors = numLayerInputs; 

	    int	halfConnectors = (numLayerConnectors - 1) / 2;

        for (int slot = 0; slot < numFncInLayer; slot++) {
            unsigned char    result = 0;
            GENOM_ITEM_TYPE   connect = 0;
            int     connectOffset = 0;
            int     stopBit = 0;
            
			connectOffset = slot - halfConnectors;	// connectors are relative, centered on current slot
			stopBit = numLayerConnectors;

            // ANALYZE ONLY SUCH NODES THAT ARE ALREADY IN USED SET
			if (displayNodes[offsetFNC + slot] == 1) {
				// COMPUTE RANGE OF INPUTS FOR PARTICULAR slot FUNCTION
				connect = genome.gene(offsetCON + slot);

				for (int bit = 0; bit < stopBit; bit++) {
					if (HasConnection(genome.gene(offsetFNC + slot), connect, slot, connectOffset, bit)) {
						if (layer > 1) {
                            int prevOffsetFNC = (layer - 2) * pGlobals->settings->circuit.sizeLayer;
							// ADD PREVIOUS NODE 	
							int targetSlot = getTargetSlot(connectOffset, bit, numLayerInputs);
							displayNodes[prevOffsetFNC + targetSlot] = 1;
//							displayNodes[prevOffsetFNC + connectOffset + bit] = 1;
						}
					}		
				}
			}
		}
	}

	return status;
}

int CircuitGenome::FilterEffectiveConnections(GENOM_ITEM_TYPE functionID, GENOM_ITEM_TYPE connectionMask, int numLayerConnectors, GENOM_ITEM_TYPE* pEffectiveConnectionMask) {
	int	status = STAT_OK;

	*pEffectiveConnectionMask = 0;

	switch (GET_FNC_TYPE(functionID)) {
		// FUNCTIONS WITH ONLY ONE CONNECTOR
		case FNC_NOP:  // no break
        case FNC_ROTL: // no break
        case FNC_ROTR: // no break
        case FNC_BITSELECTOR: {
			// Include only first connector bit
			for (int i = 0; i < numLayerConnectors; i++) {
				if (connectionMask & (GENOM_ITEM_TYPE) pGlobals->precompPow[i])  {
					*pEffectiveConnectionMask =  pGlobals->precompPow[i];
					break;
				}
			}
			break;
        }
		// FUNCTIONS WITH NO CONNECTOR
        case FNC_CONST: // no break
		case FNC_READX: *pEffectiveConnectionMask = 0; break;
		// ALL OTHER FUNCTIONS
		default: {
			// Include only bits up to numLayerConnectors
			for (int i = 0; i < numLayerConnectors; i++) {
				if (connectionMask & (GENOM_ITEM_TYPE) pGlobals->precompPow[i])  {
					*pEffectiveConnectionMask +=  pGlobals->precompPow[i];
				}
			}
			break;
        }
    }

	return status;
}

int CircuitGenome::HasConnection(GENOM_ITEM_TYPE functionID, GENOM_ITEM_TYPE connectionMask, int fncSlot, int connectionOffset, int bit) {
    int    bHasConnection = FALSE; // default is NO
    
    // DEFAULT: IF SIGNALIZED IN CONNECTOR MASK, THAN ALLOW CONNECTION
    // SOME INSTRUCTION MAY CHANGE LATER
    if (connectionMask & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) bHasConnection = TRUE;
	else bHasConnection = FALSE;
	
    switch (GET_FNC_TYPE(functionID)) {
        // 
		// FUNCTIONS WITH ONLY ONE CONNECTOR
		//
		case FNC_NOP:  // no break
        case FNC_ROTL: // no break
        case FNC_ROTR: // no break
        case FNC_BITSELECTOR: {
			// Check if this connection is the first one 
			// If not, then set connection flag bHasConnection to FALSE
			for (int i = 0; i < bit; i++) {
				if (connectionMask & (GENOM_ITEM_TYPE) pGlobals->precompPow[i]) bHasConnection = FALSE;
			}
			break;
        }
        // 
		// FUNCTIONS WITH NO CONNECTOR
		//
        case FNC_CONST: // no break
		case FNC_READX: bHasConnection = FALSE; break;
        default: {
			// connection from mask was alreadu used for all other functions
        }
    }
    
    return bHasConnection;
}

int CircuitGenome::IsOperand(GENOM_ITEM_TYPE functionID, GENOM_ITEM_TYPE connectionMask, int fncSlot, int connectionOffset, int bit, string* pOperand) {
    int    bHasConnection = HasConnection(functionID, connectionMask, fncSlot, connectionOffset, bit);
    
    switch (GET_FNC_TYPE(functionID)) {
        case FNC_NOP:	*pOperand = ""; break;
        case FNC_OR:	*pOperand = "|"; break;    
        case FNC_AND:	*pOperand = "&"; break;    
        case FNC_XOR:	*pOperand = "^"; break;    
        case FNC_NOR:	*pOperand = "| ~"; break;    
        case FNC_NAND:	*pOperand = "& ~"; break;    
        case FNC_SUM:	*pOperand = "+"; break;
        case FNC_SUBS:	*pOperand = "-"; break;     
        case FNC_ADD:	*pOperand = "+"; break;    
        case FNC_MULT:	*pOperand = "*"; break;    
        case FNC_DIV:	*pOperand = "/"; break;     
		case FNC_EQUAL: *pOperand = "=="; break;     

        case FNC_CONST: // no break
		case FNC_READX: {
			if (bHasConnection) {
				std::stringstream out;
				out << GET_FNC_ARGUMENT1(functionID);
				*pOperand = out.str();
			}
			break;
        }

        case FNC_ROTL: // no break
        case FNC_ROTR: {
			if (bHasConnection) {
				std::stringstream out;
				unsigned char tmp = GET_FNC_ARGUMENT1(functionID);
				if (GET_FNC_TYPE(functionID) == FNC_ROTL) out << "<< " << (GET_FNC_ARGUMENT1(functionID) & 0x07);
				if (GET_FNC_TYPE(functionID) == FNC_ROTR) out << ">> " << (GET_FNC_ARGUMENT1(functionID) & 0x07);
				*pOperand = out.str();
			}
			break;
        }
        case FNC_BITSELECTOR: {
			if (bHasConnection) {
				std::stringstream out;
				out << " & " << (GET_FNC_ARGUMENT1(functionID) & 0xff);
				*pOperand = out.str();
			}
			break;
        }

        default: {
            *pOperand = "!!!";
            bHasConnection = TRUE;
        }
    }
    
    return bHasConnection;
}

int CircuitGenome::GetNeutralValue(GENOM_ITEM_TYPE functionID, string* pOperand) {
    int    status = STAT_OK;
    
    switch (GET_FNC_TYPE(functionID)) {
        case FNC_OR: // no break
        case FNC_XOR: 
        case FNC_NAND: 
        case FNC_SUBS: 
        case FNC_SUM: 
        case FNC_ADD: {
            *pOperand = "0"; break;    
        }
        
        case FNC_MULT:
        case FNC_DIV: {
            *pOperand = "1"; break;    
        }

        case FNC_AND: 
        case FNC_NOR: { 
            *pOperand = "0xff"; break;    
        }

        case FNC_NOP: 
        case FNC_CONST: 
        case FNC_ROTL: 
        case FNC_BITSELECTOR: 
		case FNC_READX: 
		case FNC_EQUAL:
        case FNC_ROTR: {
            *pOperand = ""; break;    
        }

        default: {
            *pOperand = "!!!";
            status = TRUE;
        }
    }

    return status;
}

int CircuitGenome::readGenomeFromBinary(string textCircuit, GA1DArrayGenome<GENOM_ITEM_TYPE>* genome) {
    istringstream circuitStream(textCircuit);
    GENOM_ITEM_TYPE gene;
    for (int offset = 0; offset < pGlobals->settings->circuit.genomeSize; offset++) {
        circuitStream >> gene;
        if (circuitStream.fail()) {
            mainLogger.out(LOGGER_ERROR) << "Cannot load binary genome - error at offset " << offset << "." << endl;
            return STAT_DATA_CORRUPTED;
        }
        genome->gene(offset, gene);
    }
    return STAT_OK;
}

// TODO/TBD change according to printcircuit
int CircuitGenome::readGenomeFromText(string textCircuit, GA1DArrayGenome<GENOM_ITEM_TYPE>* genome) {
    GENOM_ITEM_TYPE* circuit = new GENOM_ITEM_TYPE[pGlobals->settings->circuit.genomeSize];

    int pos = 0;
    int pos2 = 0;
    int local_numLayers = 0;
    int local_intLayerSize = 0;
    int local_outLayerSize = 0;
    while ((pos2 = textCircuit.find(";", pos)) != string::npos) {
        string line = textCircuit.substr(pos, pos2 - pos + 1);
        TrimLeadingSpaces(line);
        TrimTrailingSpaces(line);
        if (line.find("[") != string::npos) {
            // AT LEAST ONE FNC ELEMENT PRESENT => NEW layer
            int offsetCON = (local_numLayers * 2) * pGlobals->settings->circuit.sizeLayer;
            int offsetFNC = (local_numLayers * 2 + 1) * pGlobals->settings->circuit.sizeLayer;
            unsigned int pos3 = 0;
            unsigned int pos4 = 0;
            int slot = 0;
            while ((pos4 = line.find("]", pos3)) != string::npos) {
                // PARSE ELEMENTS
                string elem = line.substr(pos3, pos4 - pos3 + 1);
                TrimLeadingSpaces(elem);
                TrimTrailingSpaces(elem);

                // CONNECTOR LAYER
                GENOM_ITEM_TYPE conn = (GENOM_ITEM_TYPE) StringToDouble(elem);

                // FUNCTION
                GENOM_ITEM_TYPE fnc = FNC_NOP;
                string fncStr = elem.substr(elem.length() - 5,5);
                //string fncStr = elem.Right(5);
                if (fncStr.compare("[NOP]") == 0) fnc = FNC_NOP;
                if (fncStr.compare("[OR_]") == 0) fnc = FNC_OR;
                if (fncStr.compare("[AND]") == 0) fnc = FNC_AND;
                if (fncStr.compare("[CON]") == 0) fnc = FNC_CONST;
                if (fncStr.compare("[XOR]") == 0) fnc = FNC_XOR;
                if (fncStr.compare("[NOR]") == 0) fnc = FNC_NOR;
                if (fncStr.compare("[NAN]") == 0) fnc = FNC_NAND;
                if (fncStr.compare("[ROL]") == 0) fnc = FNC_ROTL;
                if (fncStr.compare("[ROR]") == 0) fnc = FNC_ROTR;
                if (fncStr.compare("[BSL]") == 0) fnc = FNC_BITSELECTOR;
                if (fncStr.compare("[SUM]") == 0) fnc = FNC_SUM;
                if (fncStr.compare("[SUB]") == 0) fnc = FNC_SUBS;
                if (fncStr.compare("[ADD]") == 0) fnc = FNC_ADD;
                if (fncStr.compare("[MUL]") == 0) fnc = FNC_MULT;
                if (fncStr.compare("[DIV]") == 0) fnc = FNC_DIV;
                if (fncStr.compare("[RDX]") == 0) fnc = FNC_READX;
                if (fncStr.compare("[EQL]") == 0) fnc = FNC_EQUAL;

                circuit[offsetCON + slot] = conn;
                circuit[offsetFNC + slot] = fnc;

                slot++;
                pos3 = pos4 + 2;
            }

            if (local_numLayers == 0) local_intLayerSize = slot; // NUMBER OF INTERNAL ELEMENTS IS SAME AS FOR FIRST LAYER
            local_outLayerSize = slot; // WILL CONTAIN NUMBER OF ELEMENTS IN LAST LAYER

            local_numLayers = (local_numLayers) + 1;
        }
        pos = pos2 + 1;
    }
    delete circuit;
    return STAT_OK;
}

int CircuitGenome::PrintCircuit(GAGenome &g, string filePath, unsigned char *usePredictorMask, int bPruneCircuit) {
	return PrintCircuitMemory(g, filePath + "memory", usePredictorMask,bPruneCircuit);
}


int CircuitGenome::PrintCircuitMemory(GAGenome &g, string filePath, unsigned char *usePredictorMask, int bPruneCircuit) {
    int								status = STAT_OK;
    GA1DArrayGenome<GENOM_ITEM_TYPE>&inputGenome = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) g;
    GA1DArrayGenome<GENOM_ITEM_TYPE>  genome(pGlobals->settings->circuit.genomeSize, Evaluator);
    string							message;
    string							value;
    string							value2;
    string							visualCirc = "";
    string							codeCirc = "";
	string 							actualSlotID; 
	string 							previousSlotID; 
	int								bCodeCircuit = TRUE; 
    unsigned char*					displayNodes = NULL;

    displayNodes = new unsigned char[pGlobals->settings->circuit.genomeSize];

	// 1. Prune circuit (if required)
	// 2. Create header (DOT, C)
	// 3. 



    //
    // PRUNE CIRCUIT IF REQUIRED
    //
    if (pGlobals->settings->circuit.allowPrunning && bPruneCircuit) {
        // PRUNE
        status = PruneCircuit(inputGenome, genome);    
        bCodeCircuit = TRUE;

		// COMPUTE NODES TO DISPLAY 
        memset(displayNodes, 0, pGlobals->settings->circuit.genomeSize);
		status = GetUsedNodes(genome, usePredictorMask, displayNodes);        
    }
    else {
        // JUST COPY
        for (int i = 0; i < genome.size(); i++) {
			genome.gene(i, inputGenome.gene(i)); 
		}
		// ALL NODES WILL BE DISPLAYED
        memset(displayNodes, 1, pGlobals->settings->circuit.genomeSize);
    }
    
	// Compute real size of input (in case when memory is used)
	//int numberOfCircuitInputs = pGlobals->settings->circuit.sizeInputLayer;

	int numMemoryOutputs = (pGlobals->settings->circuit.useMemory) ? pGlobals->settings->circuit.memorySize : 0;
    
	//bCodeCircuit = FALSE; // BUGBUG: enable bCodeCircuit = TRUE

    // VISUAL CIRC: INPUTS 
    visualCirc += "digraph EACircuit {\r\n\
rankdir=BT;\r\n\
edge [dir=none];\r\n\
size=\"6,6\";\r\n\
ordering=out;\r\n";

	// CODE CIRCUIT: 
    if (bCodeCircuit) {
		// FUNCTION HEADER
        codeCirc += "int headerCircuit_inputLayerSize = " + toString(pGlobals->settings->circuit.sizeInputLayer) + ";\n"; 
        codeCirc += "int headerCircuit_outputLayerSize = " + toString(pGlobals->settings->circuit.totalSizeOutputLayer) + ";\n";
        codeCirc += "\n";
        codeCirc += "static void circuit(unsigned char inputs[";
        codeCirc += toString(pGlobals->settings->testVectors.inputLength);
        codeCirc += "], unsigned char outputs[";
        codeCirc += toString(pGlobals->settings->circuit.sizeOutputLayer);
        codeCirc += "]) {\n";

		// MEMORY INPUTS (if used)
		if (pGlobals->settings->circuit.useMemory) {
			int sectorLength = pGlobals->settings->circuit.sizeInputLayer - pGlobals->settings->circuit.memorySize;
			int numSectors = pGlobals->settings->testVectors.inputLength / sectorLength;

			codeCirc += "    const int SECTOR_SIZE = ";
			codeCirc += toString(sectorLength);
			codeCirc += ";\n";
			codeCirc += "    const int NUM_SECTORS = ";
			codeCirc += toString(numSectors);
			codeCirc += ";\n\n";
		}
    }

    message += "\r\n";
    for (int layer = 1; layer < 2 * pGlobals->settings->circuit.numLayers; layer = layer + 2) {
        int offsetCON = (layer-1) * pGlobals->settings->circuit.sizeLayer;
        int offsetFNC = (layer) * pGlobals->settings->circuit.sizeLayer;

        int numLayerInputs = 0;
        if (layer == 1) {
			//
			// DRAW NODES IN INPUT LAYER
			//

			// Use magenta color for memory nodes (if used)
			if (pGlobals->settings->circuit.useMemory) visualCirc += "node [color=magenta, style=filled];\r\n";

			for (int i = 0; i < pGlobals->settings->circuit.sizeInputLayer; i++) {
				// set color for data input nodes, when necessary
				if ((pGlobals->settings->circuit.useMemory && (i ==  pGlobals->settings->circuit.memorySize)) ||  // all memory inputs already processed
					(!pGlobals->settings->circuit.useMemory && (i == 0))) {										  // no memory inputs used	
					visualCirc += "node [color=green, style=filled];\r\n";
				}

				ostringstream os1;
				os1 << "IN_" << i;
				actualSlotID = os1.str();
				ostringstream os2;
				os2 << "\"" << actualSlotID << "\";\r\n";
				value2 = os2.str();
				visualCirc += value2;
				
				if (bCodeCircuit) {
					ostringstream os3;
					if (pGlobals->settings->circuit.useMemory) {
						// Initialize all inputs by zero
						os3 << "    unsigned char VAR_" << actualSlotID << " = 0;\n";
					}
					else {
						// circuit without memory, initialize inputs directly by data inputs
						os3 << "        unsigned char VAR_" << actualSlotID << " = inputs[" << i << "];\n";
					}
					value2 = os3.str();
					codeCirc += value2;
				}
			}
    		codeCirc += "\n";

			// add cycling and initialization of inputs when memory circuit is used
     		if (pGlobals->settings->circuit.useMemory) {
				codeCirc += "    for (int sector = 0; sector < NUM_SECTORS; sector++) {\n";

				// set values from input data into respective inputs 
				// NOTE: memory inputs are initialized at the end of cycle
				for (int dataInput = pGlobals->settings->circuit.memorySize; dataInput < pGlobals->settings->circuit.sizeInputLayer; dataInput++) {
					ostringstream os3;
					os3 << "        VAR_" <<  "IN_" << dataInput << " = inputs[sector * SECTOR_SIZE + " << dataInput - pGlobals->settings->circuit.memorySize << "];\n";
					value2 = os3.str();
					codeCirc += value2;
				}

				codeCirc += "\n";
			}

			numLayerInputs = pGlobals->settings->circuit.sizeInputLayer;
        }
        else numLayerInputs = pGlobals->settings->circuit.sizeLayer;

		visualCirc += "node [color=lightblue2, style=filled];\r\n";

        int numFncs = pGlobals->settings->circuit.sizeLayer;
        // IF DISPLAYING THE LAST LAYER, THEN DISPLAY ONLY 'INTERNAL_LAYER_SIZE' FNC (OTHERS ARE UNUSED)
        if (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) numFncs = pGlobals->settings->circuit.totalSizeOutputLayer;

		//
		// VISUAL CIRC: PUT ALL NODES FROM SAME LAYER INTO SAME RANK
		//
        value2 = "{ rank=same; ";
        for (int slot = 0; slot < numFncs; slot++) {
            // USE ONLY GENES THAT WERE NOT PRUNNED OUT
            if (displayNodes[offsetFNC + slot] == 1) {
			    GetFunctionLabel(genome.gene(offsetFNC + slot), genome.gene(offsetCON + slot), &value);
			    //actualSlotID.Format("\"%d_%d_%s\"; ", layer / 2 + 1, slot, value);
				ostringstream os4;
				os4 << "\"" << (layer / 2 + 1) << "_" << slot << "_" << value << "\"; ";
				actualSlotID = os4.str();
                value2 += actualSlotID; 
            }
		}
		value2 += "}\r\n";
        visualCirc += value2;

        // DISCOVER AND DRAW CONNECTIONS
        for (int slot = 0; slot < numFncs; slot++) {
			GetFunctionLabel(genome.gene(offsetFNC + slot), genome.gene(offsetCON + slot), &value);
            
            // ORDINARY LAYERS HAVE SPECIFIED NUMBER SETTINGS_CIRCUIT::numConnectors
            int	numLayerConnectors = pGlobals->settings->circuit.numConnectors;
			// IN_SELECTOR_LAYER HAS FULL INTERCONNECTION (CONNECTORS)
            if (layer / 2 == 0) numLayerConnectors = numLayerInputs;    
			// OUT_SELECTOR_LAYER HAS FULL INTERCONNECTION (CONNECTORS)
			if (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) numLayerConnectors = numLayerInputs;    
			
			int	halfConnectors = (numLayerConnectors - 1) / 2;

			GENOM_ITEM_TYPE effectiveCon = genome.gene(offsetCON + slot);
			//FilterEffectiveConnections(genome.gene(offsetFNC + slot), genome.gene(offsetCON + slot), numLayerConnectors, &effectiveCon);
			
			//value2.Format("%.10u[%s]  ", effectiveCon, value);
			// TXT: Transform relative connector mask into absolute mask (fixed inputs from previous layer)
			GENOM_ITEM_TYPE absoluteCon = 0;
			convertRelative2AbsolutConnectorMask(effectiveCon, slot, numLayerConnectors, numLayerInputs, &absoluteCon);
			ostringstream os5;
			os5 <<  setfill('0') << setw(10) << absoluteCon << "[";
			os5.setf(std::ios_base::left); os5 << setfill(' ') << setw(10) << value << "]  "; os5.unsetf(std::ios_base::left);
			value2 = os5.str();
			message += value2;
            
			// 
			// VISUAL CIRC: CREATE CONNECTION BETWEEN LAYERS
			//
			// USE ONLY GENES THAT WERE NOT PRUNNED OUT
            if (displayNodes[offsetFNC + slot] == 1) {
				//actualSlotID.Format("%d_%d_%s", layer / 2 + 1, slot, value);
				ostringstream os6;
				os6 << (layer / 2 + 1) << "_" << slot << "_" << value;
				actualSlotID = os6.str();

			    int connectOffset = slot - halfConnectors;	// connectors are relative, centered on current slot
			    int stopBit = numLayerConnectors;

				int    bFirstArgument = TRUE;
				int    bAtLeastOneConnection = FALSE;
			    
				// CODE CIRCUIT: 
				if (bCodeCircuit) {
					switch (GET_FNC_TYPE(genome.gene(offsetFNC + slot))) {
						case FNC_CONST: {
							//value2.Format("    BYTE VAR_%s = %u", actualSlotID, effectiveCon % UCHAR_MAX);
							ostringstream os7;
							os7 << "        unsigned char VAR_" << actualSlotID << " = " << (GET_FNC_ARGUMENT1(genome.gene(offsetFNC + slot)) & 0xff);
							value2 = os7.str();
							codeCirc += value2;
							bFirstArgument = FALSE;
							bAtLeastOneConnection = TRUE;
							break;
						}
						case FNC_NOR: {
							//value2.Format("    unsigned char VAR_%s = 0 | ~", actualSlotID);
							ostringstream os8;
							os8 << "        unsigned char VAR_" << actualSlotID << " = 0 | ~"; 
							value2 = os8.str();
							codeCirc += value2;
							bFirstArgument = FALSE;
							bAtLeastOneConnection = TRUE;
							break;
						}
						case FNC_NAND: {
							//value2.Format("    unsigned char VAR_%s = 0xff & ~ ", actualSlotID);
							ostringstream os9;
							os9 << "        unsigned char VAR_" << actualSlotID << " = 0xff & ~ "; 
							value2 = os9.str();
							codeCirc += value2;
							bFirstArgument = FALSE;
							bAtLeastOneConnection = TRUE;
							break;
						}
		                
					}
				}
			    
				if (bCodeCircuit) {
					// DIV AND SUBST HAVE PREVIOUS LAYER SLOT AS BASIC INPUT
					// USE ONLY GENES THAT WERE NOT PRUNNED OUT
					//if (HasImplicitConnection(GET_FNC_TYPE(genome.gene(offsetFNC + slot)))) {
					if (false){
						if (layer > 1) {
                            int prevOffsetFNC = (layer - 2) * pGlobals->settings->circuit.sizeLayer;
                            int prevOffsetCON = (layer - 3) * pGlobals->settings->circuit.sizeLayer;
							GetFunctionLabel(genome.gene(prevOffsetFNC + slot), genome.gene(prevOffsetCON + slot), &value);
							//previousSlotID.Format("%d_%d_%s", layer / 2, slot, value);
							ostringstream os10;
							os10 << (layer / 2) << "_" << slot << "_" << value;
							previousSlotID = os10.str();
						}
						else {
							//previousSlotID.Format("IN_%d", slot);
							ostringstream os11;
							os11 << "IN_" << slot;
							previousSlotID = os11.str();
						}
					    
						//value2.Format("    unsigned char VAR_%s =", actualSlotID);
						ostringstream os12;
						os12 << "        unsigned char VAR_" << actualSlotID << " =";
						value2 = os12.str();
						codeCirc += value2;

						ostringstream os13;
	                    					    
						switch (GET_FNC_TYPE(genome.gene(offsetFNC + slot))) {
							case FNC_NOP: os13 << " VAR_" << previousSlotID; value2 = os13.str(); break; 
							case FNC_SUBS: os13 << " VAR_" << previousSlotID << " - "; value2 = os13.str(); break; 
							case FNC_ADD: os13 << " VAR_" << previousSlotID << " + "; value2 = os13.str(); break; 
							case FNC_MULT: os13 << " VAR_" << previousSlotID << " * "; value2 = os13.str(); break; 
							case FNC_DIV: os13 << " VAR_" << previousSlotID << " / "; value2 = os13.str(); break; 
							case FNC_ROTL: os13 << " VAR_" << previousSlotID << " << " << (GET_FNC_ARGUMENT1(genome.gene(offsetFNC + slot) & 0x07)); value2 = os13.str(); break; 
							case FNC_ROTR: os13 << " VAR_" << previousSlotID << " >> " << (GET_FNC_ARGUMENT1(genome.gene(offsetFNC + slot) & 0x07)); value2 = os13.str(); break; 
							case FNC_BITSELECTOR: os13 << " VAR_" << previousSlotID + " & " << (GET_FNC_ARGUMENT1(genome.gene(offsetFNC + slot) & 0x07)); value2 = os13.str(); break; 
							case FNC_EQUAL: os13 << " VAR_" << previousSlotID << " == "; value2 = os13.str(); break; 
							case FNC_READX: {
								if (codeCirc[codeCirc.length()-1] == ']') {
									codeCirc.pop_back();
									os13 << " + VAR_" << previousSlotID << "%pGACirc->numInputs ]"; value2 = os13.str();
								}
								else {
									os13 << " inputs[ VAR_" << previousSlotID << "%pGACirc->numInputs ]"; value2 = os13.str();
								}
								break; 
							}
							default: 
								unsigned long a = genome.gene(offsetFNC + slot);
								value2 = "!!!"; 
						}
						codeCirc += value2;

						bFirstArgument = FALSE;
						bAtLeastOneConnection = FALSE;
					}
				}
			    
				for (int bit = 0; bit < stopBit; bit++) {
					// IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE INPUT
					if (HasConnection(genome.gene(offsetFNC + slot), effectiveCon, slot, connectOffset, bit)) {
						int    bExplicitConnection = TRUE;
						bAtLeastOneConnection = TRUE;
	                    
						if (layer > 1) {
                            int prevOffsetFNC = (layer - 2) * pGlobals->settings->circuit.sizeLayer;
                            int prevOffsetCON = (layer - 3) * pGlobals->settings->circuit.sizeLayer;
							getTargetSlot(connectOffset, bit, numLayerInputs);
							GetFunctionLabel(genome.gene(prevOffsetFNC + getTargetSlot(connectOffset, bit, numLayerInputs)), genome.gene(prevOffsetCON + getTargetSlot(connectOffset, bit, numLayerInputs)), &value);
							//previousSlotID.Format("%d_%d_%s", layer / 2, connectOffset + bit, value);
							ostringstream os21;
							os21 << (layer / 2) << "_" << getTargetSlot(connectOffset, bit, numLayerInputs) << "_" << value;
							previousSlotID = os21.str();
						}
						else {
							//previousSlotID.Format("IN_%d", connectOffset + bit);
							ostringstream os22;
							int targetSlot = getTargetSlot(connectOffset, bit, numLayerInputs);
							os22 << "IN_" << targetSlot;
							previousSlotID = os22.str();
						}
						if (bExplicitConnection) {
							//value2.Format("\"%s\" -> \"%s\";\r\n", actualSlotID, previousSlotID);
							ostringstream os23;
							os23 << "\"" << actualSlotID << "\" -> \"" << previousSlotID << "\";\r\n";
							value2 = os23.str();
							visualCirc += value2;
						}
	                    
						if (bCodeCircuit) {
							string operand;
							if (bFirstArgument) {
								//value2.Format("    unsigned char VAR_%s =", actualSlotID);
								ostringstream os25;
								os25 << "        unsigned char VAR_" << actualSlotID << " =";
								value2 = os25.str();
								codeCirc += value2;
								bFirstArgument = FALSE;
							}
							if (bExplicitConnection) {
								if (IsOperand(genome.gene(offsetFNC + slot), effectiveCon, slot, connectOffset, bit, &operand)) {
									if (GET_FNC_TYPE(genome.gene(offsetFNC + slot)) == FNC_DIV) {
										// SPECIAL FORMATING TO PREVENT ZERO DIVISION
										//value2.Format(" ((VAR_%s != 0) ? VAR_%s : 1) %s", previousSlotID, previousSlotID, operand);
										ostringstream os26;
										os26 << " ((VAR_" << previousSlotID << " != 0) ? VAR_" << previousSlotID << " : 1) " << operand;
										value2 = os26.str();
									}
									else {
										// NORMAL FORMATING
										//value2.Format(" VAR_%s %s", previousSlotID, operand);
										ostringstream os27;
										os27 << " VAR_" << previousSlotID << " " << operand;
										value2 = os27.str();
									}
									codeCirc += value2;
								}
							}
						}
					}
				}
	            
				if (bCodeCircuit) {
					if (bAtLeastOneConnection) {
						// ADD FINALIZING VALUE THAT IS NEUTRAL TO GIVEN OPERATION (E.G. 0 FOR XOR)
						string operand;
						GetNeutralValue(genome.gene(offsetFNC + slot), &operand);
						//value2.Format(" %s", operand);
						value2 = " " + operand; 
						codeCirc += value2;
						codeCirc += ";\n";
					}
					else {
						// NO CONNECTION TO GIVEN FUNCTION, OUTPUT ZERO
						//value2.Format("    unsigned char VAR_%s = 0", actualSlotID);
						ostringstream os28;
						os28 << "        unsigned char VAR_" << actualSlotID << " = 0";
						value2 = os28.str();
						codeCirc += value2;
						codeCirc += ";\n";
					}
				}
			}
        }
        message += ";\r\n";
    }
    
    if (bCodeCircuit) codeCirc += "\n";
    
	//
	// DRAW OUTPUT LAYER
	//

    // VISUAL CIRC: CONNECT OUTPUT LAYER
    //for (int i = 0; i < pGlobals->settings->circuit.totalSizeOutputLayer; i++) {

	if (pGlobals->settings->circuit.useMemory) visualCirc += "node [color=magenta];\r\n"; 

	// propagate memory outputs to inputs to next iteration (if required)
    if (pGlobals->settings->circuit.useMemory) {
		// set memory inputs by respective memory outputs 
		for (int memorySlot = 0; memorySlot < pGlobals->settings->circuit.memorySize; memorySlot++) {
            int prevOffsetFNC = (2 * pGlobals->settings->circuit.numLayers - 1) * pGlobals->settings->circuit.sizeLayer;
            int prevOffsetCON = (2 * pGlobals->settings->circuit.numLayers - 2) * pGlobals->settings->circuit.sizeLayer;
		    string		value;
		    GetFunctionLabel(genome.gene(prevOffsetFNC + memorySlot), genome.gene(prevOffsetCON + memorySlot), &value);
			ostringstream os30;
            os30 << (pGlobals->settings->circuit.numLayers) << "_" << memorySlot << "_" << value;
			previousSlotID = os30.str();

			ostringstream os3;
			os3 << "        VAR_" <<  "IN_" << memorySlot << " = VAR_" << previousSlotID << ";\n";
			value2 = os3.str();
			codeCirc += value2;
		}
	}

	int outputOffset = 0;
	for (int i = 0; i < pGlobals->settings->circuit.totalSizeOutputLayer; i++) {
		if (i == numMemoryOutputs) {
			visualCirc += "node [color=red];\r\n"; 
		}
/*
		if (usePredictorMask == NULL || usePredictorMask[i] == 1) {
			//value2.Format("node [color=red];\r\n");
			value2 = "node [color=red];\r\n"; 
			visualCirc += value2;
		}
		else {
			//value2.Format("node [color=lightblue2];\r\n");
			value2 = "node [color=lightblue2];\r\n"; 
			visualCirc += value2;
		}
*/		
		if (!bPruneCircuit || usePredictorMask == NULL || usePredictorMask[i] == 1) {
		    //actualSlotID.Format("%d_OUT", i);
			ostringstream os29;
			os29 << i << "_OUT";
			actualSlotID = os29.str();
            int prevOffsetFNC = (2 * pGlobals->settings->circuit.numLayers - 1) * pGlobals->settings->circuit.sizeLayer;
            int prevOffsetCON = (2 * pGlobals->settings->circuit.numLayers - 2) * pGlobals->settings->circuit.sizeLayer;
		    string		value;
		    GetFunctionLabel(genome.gene(prevOffsetFNC + i), genome.gene(prevOffsetCON + i), &value);
            //previousSlotID.Format("%d_%d_%s", pGACirc->settings->circuit.numLayers, i, value);
			ostringstream os30;
            os30 << (pGlobals->settings->circuit.numLayers) << "_" << i << "_" << value;
			previousSlotID = os30.str();
		    //value2.Format("\"%s\" -> \"%s\";\r\n", actualSlotID, previousSlotID);
			ostringstream os31;
			os31 << "\"" << actualSlotID << "\" -> \"" << previousSlotID << "\";\r\n";
			value2 = os31.str();
		    visualCirc += value2;
		    
            if (bCodeCircuit) {
			    //value2.Format("    outputs[%d] = VAR_%s;\r\n", i, previousSlotID);
				ostringstream os32;
			    if ((!pGlobals->settings->circuit.useMemory) || (pGlobals->settings->circuit.useMemory && i >= numMemoryOutputs)) {
					os32 << "        outputs[" << outputOffset << "] = VAR_" << previousSlotID << ";\n";
					outputOffset++;
				}
				value2 = os32.str();
			    codeCirc += value2;
            }
		}
    }
	// end parentices to cycle with memory (if required)
    if (bCodeCircuit && pGlobals->settings->circuit.useMemory) codeCirc += "    }\n";
    
	if (bCodeCircuit) codeCirc += "\n}";
    message += "\r\n";
    visualCirc += "}";

	//
    //	ACTUAL WRITING TO DISK
	//
    if (filePath == "") filePath = FILE_BEST_CIRCUIT;

	fstream	file;
	string	newFilePath;

    // WRITE FINAL: TEXT CIRCUIT
	newFilePath = filePath + ".txt";
	file.open(newFilePath.c_str(), fstream::in | fstream::out | fstream::ate | fstream::trunc);
	if (file.is_open()) {
		file << message;
		file.close();
	}

    // WRITE FINAL: BINARY GENOME (POPULATION)
    status = saveCircuitAsPopulation(genome,filePath + ".xml");

    // WRITE FINAL: C FILE
	if (bCodeCircuit) {
	    //newFilePath.Format("%s.c", filePath);
		newFilePath = filePath + ".c";
		file.open(newFilePath.c_str(), fstream::in | fstream::out | fstream::ate | fstream::trunc);
		if (file.is_open()) {
		    file << codeCirc;
		    file.close();
	    }
    }

    // WRITE FINAL: GRAPHVIZ FILE
    newFilePath = filePath + ".dot";
    file.open(newFilePath.c_str(), fstream::in | fstream::out | fstream::ate | fstream::trunc);
    if (file.is_open()) {
        file << visualCirc;
        file.close();
    }

    // DRAW CIRCUIT, IF DOT IS INSTALLED
    /*string cmdLine;
    string pngFilePath;
    //pngFilePath.Format("%s.png", filePath);
    pngFilePath = filePath + ".png";
    //cmdLine.Format("dot -Tpng %s -o %s -Gsize=1000", newFilePath, pngFilePath);
    cmdLine = "dot -Tpng " + newFilePath + " -o " + pngFilePath + " -Gsize=1000";
    WinExec(cmdLine, 0);*/

    delete[] displayNodes;

    return status;
}

int CircuitGenome::PrintCircuitMemory_DOT(GAGenome &g, string filePath, unsigned char* displayNodes) {
    int								status = STAT_OK;
    GA1DArrayGenome<GENOM_ITEM_TYPE>&genome = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) g;
//    string							message;
    string							value;
    string							value2;
    string							visualCirc = "";
//    string							codeCirc = "";
	string 							actualSlotID; 
	string 							previousSlotID; 
	int								bCodeCircuit = TRUE; 

	int numMemoryOutputs = (pGlobals->settings->circuit.useMemory) ? pGlobals->settings->circuit.memorySize : 0;
    
    // VISUAL CIRC: INPUTS 
    visualCirc += "digraph EACircuit {\r\n\
rankdir=BT;\r\n\
edge [dir=none];\r\n\
size=\"6,6\";\r\n\
ordering=out;\r\n";

    for (int layer = 1; layer < 2 * pGlobals->settings->circuit.numLayers; layer = layer + 2) {
        int offsetCON = (layer-1) * pGlobals->settings->circuit.sizeLayer;
        int offsetFNC = (layer) * pGlobals->settings->circuit.sizeLayer;

        int numLayerInputs = 0;
        if (layer == 1) {
			//
			// DRAW NODES IN INPUT LAYER
			//

			// Use magenta color for memory nodes (if used)
			if (pGlobals->settings->circuit.useMemory) visualCirc += "node [color=magenta, style=filled];\r\n";

			for (int i = 0; i < pGlobals->settings->circuit.sizeInputLayer; i++) {
				// set color for data input nodes, when necessary
				if ((pGlobals->settings->circuit.useMemory && (i ==  pGlobals->settings->circuit.memorySize)) ||  // all memory inputs already processed
					(!pGlobals->settings->circuit.useMemory && (i == 0))) {										  // no memory inputs used	
					visualCirc += "node [color=green, style=filled];\r\n";
				}

				ostringstream os1;
				os1 << "IN_" << i;
				actualSlotID = os1.str();
				ostringstream os2;
				os2 << "\"" << actualSlotID << "\";\r\n";
				value2 = os2.str();
				visualCirc += value2;
			}
			numLayerInputs = pGlobals->settings->circuit.sizeInputLayer;
        }
        else numLayerInputs = pGlobals->settings->circuit.sizeLayer;

		visualCirc += "node [color=lightblue2, style=filled];\r\n";

        int numFncs = pGlobals->settings->circuit.sizeLayer;
        // IF DISPLAYING THE LAST LAYER, THEN DISPLAY ONLY 'INTERNAL_LAYER_SIZE' FNC (OTHERS ARE UNUSED)
        if (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) numFncs = pGlobals->settings->circuit.totalSizeOutputLayer;

		//
		// VISUAL CIRC: PUT ALL NODES FROM SAME LAYER INTO SAME RANK
		//
        value2 = "{ rank=same; ";
        for (int slot = 0; slot < numFncs; slot++) {
            // USE ONLY GENES THAT WERE NOT PRUNNED OUT
            if (displayNodes[offsetFNC + slot] == 1) {
			    GetFunctionLabel(genome.gene(offsetFNC + slot), genome.gene(offsetCON + slot), &value);
			    //actualSlotID.Format("\"%d_%d_%s\"; ", layer / 2 + 1, slot, value);
				ostringstream os4;
				os4 << "\"" << (layer / 2 + 1) << "_" << slot << "_" << value << "\"; ";
				actualSlotID = os4.str();
                value2 += actualSlotID; 
            }
		}
		value2 += "}\r\n";
        visualCirc += value2;

        // DISCOVER AND DRAW CONNECTIONS
        for (int slot = 0; slot < numFncs; slot++) {
			GetFunctionLabel(genome.gene(offsetFNC + slot), genome.gene(offsetCON + slot), &value);
            
            // ORDINARY LAYERS HAVE SPECIFIED NUMBER SETTINGS_CIRCUIT::numConnectors
            int	numLayerConnectors = pGlobals->settings->circuit.numConnectors;
			// IN_SELECTOR_LAYER HAS FULL INTERCONNECTION (CONNECTORS)
            if (layer / 2 == 0) numLayerConnectors = numLayerInputs;    
			// OUT_SELECTOR_LAYER HAS FULL INTERCONNECTION (CONNECTORS)
			if (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) numLayerConnectors = numLayerInputs;    
			
			int	halfConnectors = (numLayerConnectors - 1) / 2;

			GENOM_ITEM_TYPE effectiveCon = genome.gene(offsetCON + slot);
			
			// 
			// VISUAL CIRC: CREATE CONNECTION BETWEEN LAYERS
			//
			// USE ONLY GENES THAT WERE NOT PRUNNED OUT
            if (displayNodes[offsetFNC + slot] == 1) {
				//actualSlotID.Format("%d_%d_%s", layer / 2 + 1, slot, value);
				ostringstream os6;
				os6 << (layer / 2 + 1) << "_" << slot << "_" << value;
				actualSlotID = os6.str();

			    int connectOffset = slot - halfConnectors;	// connectors are relative, centered on current slot
			    int stopBit = numLayerConnectors;

				int    bFirstArgument = TRUE;
				int    bAtLeastOneConnection = FALSE;
		    
				for (int bit = 0; bit < stopBit; bit++) {
					// IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE INPUT
					if (HasConnection(genome.gene(offsetFNC + slot), effectiveCon, slot, connectOffset, bit)) {
						int    bExplicitConnection = TRUE;
						bAtLeastOneConnection = TRUE;
	                    
						if (layer > 1) {
                            int prevOffsetFNC = (layer - 2) * pGlobals->settings->circuit.sizeLayer;
                            int prevOffsetCON = (layer - 3) * pGlobals->settings->circuit.sizeLayer;
							getTargetSlot(connectOffset, bit, numLayerInputs);
							GetFunctionLabel(genome.gene(prevOffsetFNC + getTargetSlot(connectOffset, bit, numLayerInputs)), genome.gene(prevOffsetCON + getTargetSlot(connectOffset, bit, numLayerInputs)), &value);
							//previousSlotID.Format("%d_%d_%s", layer / 2, connectOffset + bit, value);
							ostringstream os21;
							os21 << (layer / 2) << "_" << getTargetSlot(connectOffset, bit, numLayerInputs) << "_" << value;
							previousSlotID = os21.str();
						}
						else {
							//previousSlotID.Format("IN_%d", connectOffset + bit);
							ostringstream os22;
							int targetSlot = getTargetSlot(connectOffset, bit, numLayerInputs);
							os22 << "IN_" << targetSlot;
							previousSlotID = os22.str();
						}
						if (bExplicitConnection) {
							//value2.Format("\"%s\" -> \"%s\";\r\n", actualSlotID, previousSlotID);
							ostringstream os23;
							os23 << "\"" << actualSlotID << "\" -> \"" << previousSlotID << "\";\r\n";
							value2 = os23.str();
							visualCirc += value2;
						}
					}
				}
	            
			}
        }
    }
    
	//
	// DRAW OUTPUT LAYER
	//

    // VISUAL CIRC: CONNECT OUTPUT LAYER
    //for (int i = 0; i < pGlobals->settings->circuit.totalSizeOutputLayer; i++) {

	if (pGlobals->settings->circuit.useMemory) visualCirc += "node [color=magenta];\r\n"; 

	// propagate memory outputs to inputs to next iteration (if required)
    if (pGlobals->settings->circuit.useMemory) {
		// set memory inputs by respective memory outputs 
		for (int memorySlot = 0; memorySlot < pGlobals->settings->circuit.memorySize; memorySlot++) {
            int prevOffsetFNC = (2 * pGlobals->settings->circuit.numLayers - 1) * pGlobals->settings->circuit.sizeLayer;
            int prevOffsetCON = (2 * pGlobals->settings->circuit.numLayers - 2) * pGlobals->settings->circuit.sizeLayer;
		    string		value;
		    GetFunctionLabel(genome.gene(prevOffsetFNC + memorySlot), genome.gene(prevOffsetCON + memorySlot), &value);
			ostringstream os30;
            os30 << (pGlobals->settings->circuit.numLayers) << "_" << memorySlot << "_" << value;
			previousSlotID = os30.str();
		}
	}

	int outputOffset = 0;
	for (int i = 0; i < pGlobals->settings->circuit.totalSizeOutputLayer; i++) {
		if (i == numMemoryOutputs) {
			visualCirc += "node [color=red];\r\n"; 
		}
		
		if (true) {
		    //actualSlotID.Format("%d_OUT", i);
			ostringstream os29;
			os29 << i << "_OUT";
			actualSlotID = os29.str();
            int prevOffsetFNC = (2 * pGlobals->settings->circuit.numLayers - 1) * pGlobals->settings->circuit.sizeLayer;
            int prevOffsetCON = (2 * pGlobals->settings->circuit.numLayers - 2) * pGlobals->settings->circuit.sizeLayer;
		    string		value;
		    GetFunctionLabel(genome.gene(prevOffsetFNC + i), genome.gene(prevOffsetCON + i), &value);
            //previousSlotID.Format("%d_%d_%s", pGACirc->settings->circuit.numLayers, i, value);
			ostringstream os30;
            os30 << (pGlobals->settings->circuit.numLayers) << "_" << i << "_" << value;
			previousSlotID = os30.str();
		    //value2.Format("\"%s\" -> \"%s\";\r\n", actualSlotID, previousSlotID);
			ostringstream os31;
			os31 << "\"" << actualSlotID << "\" -> \"" << previousSlotID << "\";\r\n";
			value2 = os31.str();
		    visualCirc += value2;
		}
    }
    visualCirc += "}";

	//
    //	ACTUAL WRITING TO DISK
	//
    if (filePath == "") filePath = FILE_BEST_CIRCUIT;

	fstream	file;
	string	newFilePath;


    // WRITE FINAL: GRAPHVIZ FILE
    newFilePath = filePath + ".dot";
    file.open(newFilePath.c_str(), fstream::in | fstream::out | fstream::ate | fstream::trunc);
    if (file.is_open()) {
        file << visualCirc;
        file.close();
    }

    return status;
}

int CircuitGenome::ExecuteCircuit(GA1DArrayGenome<GENOM_ITEM_TYPE>* pGenome, unsigned char* inputs, unsigned char* outputs) {
    int     status = STAT_OK;
//    unsigned char*   inputsBegin = inputs;
    int     numSectors = 1;
    int     sectorLength = pGlobals->settings->circuit.sizeInputLayer;
    int     memoryLength = 0;
    unsigned char*    localInputs = NULL;
    unsigned char*    localOutputs = NULL;
    unsigned char*    fullLocalInputs = NULL;

	// Compute maximum number of inputs into any layer
	int maxLayerSize = (pGlobals->settings->circuit.sizeInputLayer > pGlobals->settings->circuit.sizeLayer) ? pGlobals->settings->circuit.sizeInputLayer : pGlobals->settings->circuit.sizeLayer;

	// Local inputs and local outputs contains inputs and outputs for particular layer
	// TRICK: We will use relative connectors which may request to access input values from other side of input array (connectors for slot 0 can access slot -2 == sizeLayer - 2)
	// To remove necessity for repeated boundary checks, we will duplicite inputs three times in the following pattern:
	// in1, in2, in3 ... inx | in1, in2, in3 ... inx | in1, in2, in3 ... inx
	//                         ^
	//						   | localInputs pointer => localInputs[-2] will access inx-1;					
    fullLocalInputs = new unsigned char[3*maxLayerSize];
	localInputs = fullLocalInputs + maxLayerSize;	// set localInputs ptr to point at the middle occurence of pattern
    localOutputs = new unsigned char[maxLayerSize];
    memset(fullLocalInputs, 0, 3*maxLayerSize);
    memset(localOutputs, 0, maxLayerSize);

	if (pGlobals->settings->circuit.useMemory) {
		// USE MEMORY WITH PARTITIONED INPUTS

		// | MEMORY_INPUTS_i | DATA_INPUTS_i |
		// ######### CIRCUIT #############
		// | MEMORY_INTPUTS_i+1 | DATA_OUTPUTS_i |
		// 
		// sizeof(MEMORY_INPUTS_i) == pGlobals->settings->circuit.memorySize
		// sizeof(DATA_INPUTS_i) == pGlobals->settings->circuit.sizeInputLayer - pGlobals->settings->circuit.memorySize
		// sizeof(MEMORY_INTPUTS_i+1) == pGlobals->settings->circuit.memorySize
		// sizeof(DATA_OUTPUTS_i) == pGlobals->settings->circuit.sizeOutputLayer

		sectorLength = pGlobals->settings->circuit.sizeInputLayer - pGlobals->settings->circuit.memorySize;
		memoryLength = pGlobals->settings->circuit.memorySize;
		numSectors = pGlobals->settings->testVectors.inputLength / sectorLength;

		assert(pGlobals->settings->testVectors.inputLength % sectorLength == 0);
	}
	else {
		// ALL IN ONE RUN
		numSectors = 1;
		memoryLength = 0;
		sectorLength = pGlobals->settings->circuit.sizeInputLayer;
	}

    for (int sector = 0; sector < numSectors; sector++) { 
        // PREPARE INPUTS FOR ACTUAL RUN OF CIRCUIT
        if (numSectors == 1) {
            // ALL INPUT DATA AT ONCE
			memcpy(localInputs, inputs, pGlobals->settings->circuit.sizeInputLayer);
			// duplicate before and after (see TRICK above)
			memcpy(localInputs - pGlobals->settings->circuit.sizeInputLayer, localInputs, pGlobals->settings->circuit.sizeInputLayer);
			memcpy(localInputs + pGlobals->settings->circuit.sizeInputLayer, localInputs, pGlobals->settings->circuit.sizeInputLayer);
        }
        else {
            // USE MEMORY STATE (OUTPUT) AS FIRST PART OF INPUT
			// NOTE: for first iteration, memory is zero (taken from localOutputs)
            memcpy(localInputs, localOutputs, memoryLength);
            // ADD FRESH INPUT DATA
            memcpy(localInputs + memoryLength, inputs + sector * sectorLength, sectorLength);
			int realInputsLength = memoryLength + sectorLength;
			assert(realInputsLength <= maxLayerSize);
			// duplicate before and after
			memcpy(localInputs - realInputsLength, localInputs, realInputsLength);
			memcpy(localInputs + realInputsLength, localInputs, realInputsLength);
        }
        
        // EVALUATE CIRCUIT
        for (int layer = 1; layer < 2 * pGlobals->settings->circuit.numLayers; layer = layer + 2) {
            int offsetCON = (layer-1) * pGlobals->settings->circuit.sizeLayer;
            int offsetFNC = (layer) * pGlobals->settings->circuit.sizeLayer;
            memset(localOutputs, 0, maxLayerSize); // BUGBUG: can be sizeInputLayer lower than number of used items in localOutputs?

            // actual number of inputs for this layer. For first layer equal to pCircuit->numInputs, for next layers equal to number of function in intermediate layer pCircuit->internalLayerSize
            int numLayerInputs = 0;
            if (layer == 1) {
                // WHEN SECTORING IS USED THEN FIRST LAYER OBTAIN sizeInputLayer INPUTS
                numLayerInputs = pGlobals->settings->circuit.sizeInputLayer;
            }
            else numLayerInputs = pGlobals->settings->circuit.sizeLayer;

            // actual number of functions in layer - different for the last "output" layer
            int numFncInLayer = (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) ? (pGlobals->settings->circuit.totalSizeOutputLayer) : pGlobals->settings->circuit.sizeLayer;

            // ORDINARY LAYERS HAVE SPECIFIED NUMBER SETTINGS_CIRCUIT::numConnectors
            int	numLayerConnectors = pGlobals->settings->circuit.numConnectors;
			// IN_SELECTOR_LAYER HAS FULL INTERCONNECTION (CONNECTORS)
            if (layer / 2 == 0) numLayerConnectors = numLayerInputs;    
			// OUT_SELECTOR_LAYER HAS FULL INTERCONNECTION (CONNECTORS)
			if (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) numLayerConnectors = numLayerInputs;    
			// IF NUMBER OF CONNECTORS IS HIGHER THAN NUMBER OF FUNCTIONS IN LAYER => LIMIT TO numLayerInputs
			if (numLayerConnectors > numLayerInputs) numLayerConnectors = numLayerInputs; 

			// halfConnectors is used for relative interpretation of connectors
			int	halfConnectors = (numLayerConnectors - 1) / 2;

            for (int slot = 0; slot < numFncInLayer; slot++) {
                unsigned char	result = 0;
                GENOM_ITEM_TYPE   connect = 0;
                int     connectOffset = 0;
                int     stopBit = 0;
                
				//
				// COMPUTE RANGE OF INPUTS FOR PARTICULAR slot FUNCTION
				//
                connect = pGenome->gene(offsetCON + slot);

			    connectOffset = slot - halfConnectors;	// connectors are relative, centered on current slot
			    stopBit = numLayerConnectors;

				//
				// EVALUATE FUNCTION BASED ON ITS INPUTS 
				//
                switch (GET_FNC_TYPE(pGenome->gene(offsetFNC + slot))) {
                    case FNC_NOP: {
                        // DO NOTHING, JUST PASS VALUE FROM FIRST CONNECTOR FROM PREVIOUS LAYER
                        for (int bit = 0; bit < stopBit; bit++) {
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
								result = localInputs[connectOffset + bit];
								break; // pass only one value
                            }
                        }
                        break;
                    }
                    case FNC_OR: {
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE INPUT
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result |= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_AND: {
                        result = ~0; // ASSIGN ALL ONES AS STARTING VALUE TO PERFORM result &= LATER
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION, THEN TAKE INPUT
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result &= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_CONST: {
                        // SEND VALUE FROM CONNECTION LAYER DIRECTLY TO OUTPUT
                        result = GET_FNC_ARGUMENT1(pGenome->gene(offsetCON + slot));
                        break;
                    }
                    case FNC_XOR: {
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE INPUT
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result ^= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_NOR: {
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE NEGATION OF INPUT
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result |= ~(localInputs[connectOffset + bit]);
                            }
                        }
                        break;
                    }
                    case FNC_NAND: {
                        result = ~0; // ASSIGN ALL ONES AS STARTING VALUE TO PERFORM result &= LATER
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE NEGATION OF INPUT
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result &= ~(localInputs[connectOffset + bit]);
                            }
                        }
                        break;
					}
                    case FNC_ROTL: {
                        // SHIFT IS ENCODED IN FUNCTION IDENTFICATION 
						// MAXIMUM SHIFT IS 7
                        for (int bit = 0; bit < stopBit; bit++) {
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
								result = localInputs[connectOffset + bit] << (GET_FNC_ARGUMENT1(pGenome->gene(offsetFNC + slot) & 0x07));
								break; // pass only one value
                            }
                        }
                        
                        break;
                    }
                    case FNC_ROTR: {
                        // SHIFT IS ENCODED IN FUNCTION IDENTFICATION 
						// MAXIMUM SHIFT IS 7
                        for (int bit = 0; bit < stopBit; bit++) {
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
								result = localInputs[connectOffset + bit] >> (GET_FNC_ARGUMENT1(pGenome->gene(offsetFNC + slot) & 0x07));
								break; // pass only one value
                            }
                        }
                        
                    }
                    case FNC_BITSELECTOR: {
                        // BIT SELECTOR
                        // MASK IS ENCODED IN FUNCTION IDENTFICATION 
                        for (int bit = 0; bit < stopBit; bit++) {
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
								result = localInputs[connectOffset + bit] & (GET_FNC_ARGUMENT1(pGenome->gene(offsetFNC + slot) & 0x07));
								break; // pass only fisrt value
                            }
                        }
                        
                        break;
                    }
                    case FNC_SUM: {
                        // SUM ALL INPUTS
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN SUM IT INTO OF INPUT
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result += localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_SUBS: {
                        // SUBSTRACT ALL REMAINING VALUES FROM FIRST CONNECTED INPUTS
                        bool bFirstInput = true;
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN SUBSTRACT IT OF INPUT
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                if (bFirstInput) {
									result = localInputs[connectOffset + bit];
									bFirstInput = false;
								}
								else result -= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_ADD: {
                        // TAKE INPUT FROM THE FUNCTION FROM PREVIOUS LAYER ON THE SAME POSITION AS THIS ONE
                        // AND ADD ALL VALUES FROM CONNECTED INPUTS
                        result = 0;
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN ADD THIS INPUT 
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result += localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_MULT: {
                        // MULTIPLY ALL VALUES FROM CONNECTED INPUTS
                        result = 1;
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN MULTIPLY THIS INPUT
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result *= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_DIV: {
                        // DIVIDE FIRST INPUT ALL VALUES REMAINING CONNECTED INPUTS
                        bool bFirstInput = true;
                        result = 0;
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN DIVIDE THIS INPUT
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                if (bFirstInput) {
									result = localInputs[connectOffset + bit];
									bFirstInput = true;
								}
								else {
									if (localInputs[connectOffset + bit] != 0) {
										result /= localInputs[connectOffset + bit];
									}
								}
                            }
                        }
                        break;
                    }
					case FNC_READX: {
						// READ x-th byte - address X is computed as sum from all inputs (modulo size of input layer)
						result = 0;
						for (int bit = 0; bit < stopBit; bit++) {
							if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
								result += localInputs[connectOffset + bit];
							}
						}
                        result = inputs[result % pGlobals->settings->circuit.sizeInputLayer];
                        break;
					}
					case FNC_EQUAL: {
                        // COMPARE ALL INPUTS
						bool bFirstInput = true;
						unsigned char firstValue = 0;
						result = 1; // assume equalilty of inputs
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TEST EQUALITY OF INPUTS
                            if (connect & (GENOM_ITEM_TYPE) pGlobals->precompPow[bit]) {
								if (bFirstInput) {
									bFirstInput = false;
									firstValue = localInputs[connectOffset + bit];
								}
								else {
									if (firstValue != localInputs[connectOffset + bit]) {
										// inputs are not equal
										result = 0;
										break;	// stop testing
									}
								}
                            }
                        }
                        break;
					}
                    default: {
                        mainLogger.out(LOGGER_ERROR) << "Unknown function in circuit. (" << pGenome->gene(offsetFNC + slot) << ")" << endl;
                        assert(FALSE);
                        break;
                    }
				}
                
                localOutputs[slot] = result;
            }
            // PREPARE INPUTS FOR NEXT LAYER FROM OUTPUTS
            memcpy(localInputs, localOutputs, pGlobals->settings->circuit.sizeLayer);

			// duplicate before and after (see TRICK above)
			memcpy(localInputs - pGlobals->settings->circuit.sizeLayer, localInputs, pGlobals->settings->circuit.sizeLayer);
			memcpy(localInputs + pGlobals->settings->circuit.sizeLayer, localInputs, pGlobals->settings->circuit.sizeLayer);

        }
    }
    
	// circuit output is taken from output parts AFTER memory outputs
	// BUT length of memory outputs can be zero (in case of circuit without memory)
	memcpy(outputs,localOutputs + memoryLength,pGlobals->settings->circuit.sizeOutputLayer);

	delete[] fullLocalInputs;
    delete[] localOutputs;
    return status;
}

int CircuitGenome::writeGenome(const GA1DArrayGenome<GENOM_ITEM_TYPE>& genome, string& textCircuit) {
    int status = STAT_OK;

    ostringstream textCicruitStream;
    for (int i = 0; i < genome.length(); i++) {
        textCicruitStream << genome.gene(i) << " ";
        if (i % pGlobals->settings->circuit.sizeLayer == pGlobals->settings->circuit.sizeLayer - 1) {
            textCicruitStream << "  ";
        }
    }
    textCircuit = textCicruitStream.str();

    return status;
}

int CircuitGenome::saveCircuitAsPopulation(const GA1DArrayGenome<GENOM_ITEM_TYPE>& genome, const string filename) {
    int status = STAT_OK;
    TiXmlElement* pRoot = CircuitGenome::populationHeader(1);
    TiXmlElement* pElem = NULL;
    TiXmlElement* pElem2 = NULL;

    pElem = new TiXmlElement("population");
    string textCircuit;
    GA1DArrayGenome<GENOM_ITEM_TYPE>* pGenome = (GA1DArrayGenome<GENOM_ITEM_TYPE>*) &genome;
    status = CircuitGenome::writeGenome(*pGenome ,textCircuit);
    if (status != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Could not save circuit fo file " << filename << "." << endl;
        return status;
    }
    pElem2 = new TiXmlElement("genome");
    pElem2->LinkEndChild(new TiXmlText(textCircuit.c_str()));
    pElem->LinkEndChild(pElem2);
    pRoot->LinkEndChild(pElem);

    status = saveXMLFile(pRoot, filename);
    if (status != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Cannot save circuit to file " << filename << "." << endl;
        return status;
    }
    return status;
}

TiXmlElement* CircuitGenome::populationHeader(int populationSize) {
    TiXmlElement* pRoot = new TiXmlElement("eacirc_population");
    TiXmlElement* pElem = NULL;
    TiXmlElement* pElem2 = NULL;

    pElem = new TiXmlElement("population_size");
    pElem->LinkEndChild(new TiXmlText(toString(populationSize).c_str()));
    pRoot->LinkEndChild(pElem);
    pElem = new TiXmlElement("circuit_dimensions");
    pElem2 = new TiXmlElement("num_layers");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->circuit.numLayers).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("size_layer");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->circuit.sizeLayer).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("size_input_layer");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->circuit.sizeInputLayer).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("size_output_layer");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->circuit.sizeOutputLayer).c_str()));
    pElem->LinkEndChild(pElem2);
    pRoot->LinkEndChild(pElem);

    return pRoot;
}
