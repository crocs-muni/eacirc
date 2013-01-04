#include "CircuitGenome.h"
#include "Evaluator.h"
#include "EACirc.h"
#include "CommonFnc.h"
#include "circuit_evaluator/ICircuitEvaluator.h"
#include <iomanip>

void CircuitGenome::ExecuteFromText(string textCircuit, GA1DArrayGenome<unsigned long> *genome) {
    unsigned long   circuit[MAX_GENOME_SIZE];    
    string          message;

    memset(circuit, 0, sizeof(circuit));
	ParseCircuit(textCircuit, circuit, &(pGACirc->numLayers), &(pGACirc->internalLayerSize), &(pGACirc->outputLayerSize));
    
    // ACTUAL SIZE OF GENOM
    pGACirc->genomeSize = (pGACirc->numLayers * 2) * MAX_INTERNAL_LAYER_SIZE;
    
    for (int i = 0; i < pGACirc->genomeSize; i++) genome->gene(i, circuit[i]);

//	PrintCircuit();
    
    pGACirc->clearFitnessStats();
}

float CircuitGenome::Evaluator(GAGenome &g) {
    int								status = STAT_OK;
    GA1DArrayGenome<unsigned long>  &genome = (GA1DArrayGenome<unsigned long>&) g;
    string							message;
    float							fit = 0;
    unsigned char					inputs[MAX_INPUTS];    
    unsigned char					outputs[MAX_OUTPUTS];  
    unsigned char					usePredictorMask[MAX_OUTPUTS];  
    unsigned char					correctOutputs[MAX_OUTPUTS];  
    int								match = 0;
    int								numPredictions = 0; 
	int								remainingTestVectors = pGACirc->numTestVectors;
	ICircuitEvaluator				*circEval = new ICircuitEvaluator();
    {
        ICircuitEvaluator* temp = circEval->getCircEvalClass();
        delete circEval;
        circEval = temp;
    }

	memset(usePredictorMask, 1, sizeof(usePredictorMask));	// USE ALL PREDICTORS
	remainingTestVectors = pGACirc->numTestVectors;

    match = 0;
	int match1 = 0;
    int match2 = 0;    
    numPredictions = 0;
    for (int testSet = 0; testSet < remainingTestVectors; testSet++) { 
        // LOAD TEST SET INTO inputs
		memcpy(inputs, pGACirc->testVectors[testSet], MAX_INPUTS);
        // LOAD TEST SET INTO outputs
        memcpy(correctOutputs, pGACirc->testVectors[testSet] + MAX_INPUTS, MAX_OUTPUTS);
            
        // EXECUTE CIRCUIT
        status = ExecuteCircuit(&genome, inputs, outputs);
        
        // EVALUATE SUCCESS OF CIRCUIT FOR THIS TEST VECTOR
		circEval->evaluateCircuit(outputs, correctOutputs, usePredictorMask, &match, &numPredictions);
    }

	// COMPARE CIRCUIT INPUTS AND OUTPUTS, IF SAME, SET FITNESS TO 0
	int inOutCompare = memcmp(inputs, outputs, pGACirc->outputLayerSize);
	if (inOutCompare == 0) {
		fit = 0;
	} else {
		fit = (numPredictions > 0) ? (match / ((float) numPredictions)) : 0;
	}

    if (!pGACirc->prunningInProgress) {
        // include into average fitness of whole generation
        (pGACirc->avgGenerFit) += fit;
        (pGACirc->numAvgGenerFit)++;
        (pGACirc->avgPredictions) += numPredictions;

        if (fit > pGACirc->bestGenerFit) pGACirc->bestGenerFit = fit;

        if (pGACirc->maxFit < fit) {
            pGACirc->maxFit = fit;

            // DISPLAY CURRENTLY BEST
            ostringstream os2;
            os2 << FILE_CIRCUIT << setprecision(CIRCUIT_FILENAME_PRECISION) << fixed << fit;
            string filePath = os2.str();
            PrintCircuit(genome, filePath, usePredictorMask, FALSE);   // PRINT WITHOUT PRUNNING

            if (pGACirc->allowPrunning) {
                filePath += "_prunned";
                PrintCircuit(genome, filePath, usePredictorMask, TRUE);    // PRINT WITH PRUNNING
            }
        }
    }
        
    delete circEval;
    return fit;
}

void CircuitGenome::Initializer(GAGenome &g) {
    GA1DArrayGenome<unsigned long> &genome = (GA1DArrayGenome<unsigned long>&) g;
    int	offset = 0;

    // loading done elsewhere (manually in eacirc.run after initializing GAlib)
    // LOAD AND CONTINUE
    /*
	string fileName = "EAC_circuit.bin";
	fstream	efile;
	string executetext;
	efile.open(fileName.c_str(), fstream::in);
	if (efile.is_open()) {
		getline(efile, executetext);
		CircuitGenome::ExecuteFromText(executetext, &genome);
		efile.close();
		return;
    }
    */

	// CLEAR GENOM
	for (int i = 0; i < genome.size(); i++) genome.gene(i, 0);
    
	// INITIALIZE ALL LAYERS

    for (int layer = 0; layer < 2 * pGACirc->numLayers; layer++) {
        offset = layer * MAX_INTERNAL_LAYER_SIZE; 
        // LAST LAYER CAN HAVE DIFFERENT NUMBER OF FUNCTIONS
        int numLayerInputs = 0;
        if (layer == 0) {
            // WHEN SECTORING IS USED THEN FIRST LAYER OBTAIN internalLayerSize INPUTS
            //if (pGACirc->bSectorInputData) numLayerInputs = pGACirc->internalLayerSize; 
            numLayerInputs = pGACirc->numInputs;
        }
        else numLayerInputs = pGACirc->internalLayerSize;
        
        int numFncInLayer = ((layer == 2 * pGACirc->numLayers - 1) || (layer == 2 * pGACirc->numLayers - 2)) ? pGACirc->outputLayerSize : pGACirc->internalLayerSize;

        for (int slot = 0; slot < numFncInLayer; slot++) {
            unsigned long   value;
            if (layer % 2 == 0) {
                // CONNECTION SUB-LAYER
                if (layer / 2 < pGACirc->numSelectorLayers) {
                    // SELECTOR LAYER - TAKE INPUT ONLY FROM PREVIOUS NODE IN SAME COORDINATES
                    value = pGACirc->precompPow[slot];
                }
                else {
                    // FUNCTIONAL LAYERS
                    rndGen->getRandomFromInterval(pGACirc->precompPow[numLayerInputs], &value);
                    //value = 0xffffffff;
                }
                genome.gene(offset + slot, value);
            }
            else {
                // FUNCTION SUB-LAYER, SET ONLY ALLOWED FUNCTIONS  
                if (layer / 2 < pGACirc->numSelectorLayers) {
                    // SELECTOR LAYER - PASS INPUT WITHOUT CHANGES (ONLY SIMPLE COMPOSITION OF INPUTS IS ALLOWED)
                    genome.gene(offset + slot, FNC_XOR);
                }
                else {
                    // FUNCTIONAL LAYER
                    int bFNCNotSet = TRUE;
                    while (bFNCNotSet) {
                        rndGen->getRandomFromInterval(FNC_MAX, &value);
                        if (pGACirc->allowedFNC[value] != 0) { 
                            genome.gene(offset + slot, value);
                            bFNCNotSet = FALSE;
                        }
                    }
                }
            }
        }
    }
}

int CircuitGenome::Mutator(GAGenome &g, float pmut) {
    GA1DArrayGenome<unsigned long> &genome = (GA1DArrayGenome<unsigned long>&) g;
    int result = 0;
    
    for (int layer = 0; layer < 2 * pGACirc->numLayers; layer++) {
        int offset = layer * MAX_INTERNAL_LAYER_SIZE; 

        int numLayerInputs = 0;
        if (layer == 0) {
            // WHEN SECTORING IS USED THEN FIRST LAYER OBTAIN internalLayerSize INPUTS
            //if (pGACirc->bSectorInputData) numLayerInputs = pGACirc->internalLayerSize; 
            numLayerInputs = pGACirc->numInputs;
        }
        else numLayerInputs = pGACirc->internalLayerSize;

        int numFncInLayer = ((layer == 2 * pGACirc->numLayers - 1) || (layer == 2 * pGACirc->numLayers - 2)) ? pGACirc->outputLayerSize : pGACirc->internalLayerSize;

        for (int slot = 0; slot < numFncInLayer; slot++) {
            unsigned long value;
            if (layer % 2 == 0) {
                // CONNECTION SUB-LAYER
                
                // MUTATE CONNECTION SELECTOR (FLIP ONE SINGLE BIT == ONE CONNECTOR)
                if (GAFlipCoin(pmut)) {
                    unsigned long temp;
                    rndGen->getRandomFromInterval(numLayerInputs, &value);
                    temp = pGACirc->precompPow[value];
                    // SWITCH RANDOMLY GENERATED BIT
                    temp ^= genome.gene(offset + slot);
                    genome.gene(offset + slot, temp);
                }
            }
            else {
                // MUTATE FUNCTION TYPE USING ONLY ALLOWED FNCs
                if (GAFlipCoin(pmut)) {             
                    int bFNCNotSet = TRUE;
                    while (bFNCNotSet) {
                        rndGen->getRandomFromInterval(FNC_MAX, &value);
                        if (pGACirc->allowedFNC[value] != 0) { 
                            genome.gene(offset + slot, value);
                            bFNCNotSet = FALSE;
                        }
                    }
                }
            }
        }
    }


    return result;
}

int CircuitGenome::Crossover(const GAGenome &p1, const GAGenome &p2, GAGenome *o1, GAGenome *o2) {
    GA1DArrayGenome<unsigned long> &parent1 = (GA1DArrayGenome<unsigned long>&) p1;
    GA1DArrayGenome<unsigned long> &parent2 = (GA1DArrayGenome<unsigned long>&) p2;
    GA1DArrayGenome<unsigned long> &offspring1 = (GA1DArrayGenome<unsigned long>&) *o1;
    GA1DArrayGenome<unsigned long> &offspring2 = (GA1DArrayGenome<unsigned long>&) *o2;
    
    // CROSS ONLY WHOLE LAYERS
    int cpoint = GARandomInt(1,pGACirc->numLayers) * 2; // bod ken (v mezch od,do)
    for (int layer = 0; layer < 2 * pGACirc->numLayers; layer++) {
        int offset = layer * MAX_INTERNAL_LAYER_SIZE; 

        if (layer <= cpoint) {
            for (int i = 0; i < MAX_INTERNAL_LAYER_SIZE; i++) {
                if (o1 != NULL) offspring1.gene(offset + i, parent1.gene(offset + i));
                if (o2 != NULL) offspring2.gene(offset + i, parent2.gene(offset + i));
            }
        }
        else {
            for (int i = 0; i < MAX_INTERNAL_LAYER_SIZE; i++) {
                if (o1 != NULL) offspring1.gene(offset + i, parent2.gene(offset + i));
                if (o2 != NULL) offspring2.gene(offset + i, parent1.gene(offset + i));
            }
        }
    }
    return 1;
}

int CircuitGenome::GetFunctionLabel(unsigned long functionID, unsigned long connections, string* pLabel) {
    int		status = STAT_OK;
    switch (functionID) {
        case FNC_NOP: *pLabel = "NOP"; break;
        case FNC_OR: *pLabel = "OR_"; break;
        case FNC_AND: *pLabel = "AND"; break;
//        case FNC_CONST: *pLabel = "CON"; break;
        case FNC_CONST: {
            //pLabel->Format("CONST_%u", connections % UCHAR_MAX);
			std::stringstream out;
			out << (connections % UCHAR_MAX);
			*pLabel = "CONST_" + out.str();;
            break;
        }
		case FNC_READX: *pLabel = "RDX"; break;
        case FNC_XOR: *pLabel = "XOR"; break;
        case FNC_NOR: *pLabel = "NOR"; break;
        case FNC_NAND: *pLabel = "NAN"; break;
//        case FNC_ROTL: *pLabel = "ROL"; break;
        case FNC_ROTL: {
			std::stringstream out;
			out << (connections & 0x07);
            *pLabel = "ROL_" + out.str(); 
            break;
        }
//        case FNC_ROTR: *pLabel = "ROR"; break;
        case FNC_ROTR: {
			std::stringstream out;
			out << (connections & 0x07);
            *pLabel = "ROR_" + out.str(); 
            break;
        }
//        case FNC_BITSELECTOR: *pLabel = "BSL"; break;
        case FNC_BITSELECTOR: {
			std::stringstream out;
			out << (connections & 0x07);
            *pLabel = "BSL_" + out.str(); 
            break;
        }
        case FNC_SUM: *pLabel = "SUM"; break;
        case FNC_SUBS: *pLabel = "SUB"; break;
        case FNC_ADD: *pLabel = "ADD"; break;
        case FNC_MULT: *pLabel = "MUL"; break;
        case FNC_DIV: *pLabel = "DIV"; break;
        default: {
            assert(FALSE);
            *pLabel = "ERR";
            status = STAT_USERDATA_BAD;
        }
    }
    
    return status;
}

int CircuitGenome::PruneCircuit(GAGenome &g, GAGenome &prunnedG) {
    int                     status = STAT_OK;
    GA1DArrayGenome<unsigned long>  &genome = (GA1DArrayGenome<unsigned long>&) g;
    GA1DArrayGenome<unsigned long>  &prunnedGenome = (GA1DArrayGenome<unsigned long>&) prunnedG;
    int                    bChangeDetected = FALSE;

    // CREATE LOCAL COPY
    for (int i = 0; i < genome.size(); i++) {
        prunnedGenome.gene(i, genome.gene(i)); 
    }

    if (pGACirc->prunningInProgress) { 
        // WE ARE ALREDY PERFORMING PRUNING - DO NOT CONTINUE TO PREVENT OVERLAPS
    }
    else {
        //
        // METHOD - TRY TO TEMPORARY REMOVE CONNECTION/FUNCTION AND TEST FITNESS CHANGES
        //
        
        pGACirc->prunningInProgress = TRUE;        
        
		float origFit = Evaluator(prunnedGenome);
        
        int prunneRepeat = 0; 
        bChangeDetected = TRUE;
        while (bChangeDetected && prunneRepeat < 10) {
            bChangeDetected = FALSE;
            prunneRepeat++;
            
            // DISABLE GENES STARTING FROM END 
            for (int i = prunnedGenome.size() - 1; i >= 0; i--) {
                unsigned long   origValue = prunnedGenome.gene(i);
                
                if (origValue != 0) {
                    // PRUNE FNC AND CONNECTION LAYER DIFFERENTLY
                    if (((i / MAX_INTERNAL_LAYER_SIZE) % 2) == 1) {
                        // FNCs LAYER - TRY TO SET AS NOP INSTRUCTION
                        prunnedGenome.gene(i, 0);
                        
                        assert(origValue <= FNC_MAX);
                        
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
                        unsigned long   tempOrigValue = origValue;  // WILL HOLD MASK OF INPORTANT CONNECTIONS
                        // CONNECTION LAYER - TRY TO REMOVE CONNECTIONS GRADUALLY
                        for (int conn = 0; conn < MAX_CONNECTORS; conn++) {
                            unsigned long   newValue = tempOrigValue & (~pGACirc->precompPow[conn]);
                            
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
    
    // !!! WHAT IS THIS SUPPOSSED TO BE DOING? scope start
    //Evaluator(prunnedGenome);
    // !!! WHAT IS THIS SUPPOSSED TO BE DOING? scope end

    pGACirc->prunningInProgress = FALSE;
    return status;
}

int CircuitGenome::GetUsedNodes(GAGenome &g, unsigned char usePredictorMask[MAX_OUTPUTS], unsigned char displayNodes[MAX_GENOME_SIZE]) {
	int	status = STAT_OK;
    GA1DArrayGenome<unsigned long>  &genome = (GA1DArrayGenome<unsigned long>&) g;
	
	//
	// BUILD SET OF USED NODES FROM OUTPUT TO INPUT
	//
	
	// ADD OUTPUT NODES
    // VISUAL CIRC: CONNECT OUTPUT LAYER
    int offsetFNC = (2 * pGACirc->numLayers - 1) * MAX_INTERNAL_LAYER_SIZE; 
    for (int i = 0; i < pGACirc->outputLayerSize; i++) {
		if (usePredictorMask == NULL || usePredictorMask[i] == 1) {
			// ADD THIS ONE TO LIST OF USED NODES 
			displayNodes[offsetFNC + i] = 1;	
		}
    }
	
	
	// PROCESS ALL LAYERS FROM BACK
    for (int layer = 2 * pGACirc->numLayers - 1; layer > 0; layer = layer - 2) {
        int offsetCON = (layer-1) * MAX_INTERNAL_LAYER_SIZE; 
        int offsetFNC = (layer) * MAX_INTERNAL_LAYER_SIZE; 

        // actual number of inputs for this layer. For first layer equal to pCircuit->numInputs, for next layers equal to number of function in intermediate layer pCircuit->internalLayerSize
        int numLayerInputs = 0;
        if (layer == 1) {
            // WHEN SECTORING IS USED THEN FIRST LAYER OBTAIN internalLayerSize INPUTS
            //if (pGACirc->bSectorInputData) numLayerInputs = pGACirc->internalLayerSize; 
            numLayerInputs = pGACirc->numInputs;
        }
        else numLayerInputs = pGACirc->internalLayerSize;

        // actual number of functions in layer - different for the last "output" layer
        int numFncInLayer = (layer == (2 * pGACirc->numLayers - 1)) ? pGACirc->outputLayerSize : pGACirc->internalLayerSize; 

        // SELECTOR LAYERS HAVE FULL INTERCONNECTION (CONNECTORS), OTHER HAVE SPECIFIED NUMBER 
	    int	numLayerConnectors = pGACirc->numLayerConnectors;
        if (layer / 2 < pGACirc->numSelectorLayers) {
            numLayerConnectors = numLayerInputs;    
        }
	    int	halfConnectors = (numLayerConnectors - 1) / 2;

        for (int slot = 0; slot < numFncInLayer; slot++) {
            unsigned char    result = 0;
            unsigned long   connect = 0;
            int     connectOffset = 0;
            int     stopBit = 0;
            
            // ANALYZE ONLY SUCH NODES THAT ARE ALREADY IN USED SET
			if (displayNodes[offsetFNC + slot] == 1) {
				// COMPUTE RANGE OF INPUTS FOR PARTICULAR slot FUNCTION
				connect = genome.gene(offsetCON + slot);
				
				if (numLayerConnectors > numLayerInputs) {
					// NUMBER OF CONNECTORS IS HIGHER THAN NUMBER OF FUNCTIONS IN LAYER - CUT ON BOTH SIDES			
					connectOffset = 0;
					stopBit = numLayerInputs;
				}
				else {
					connectOffset = slot - halfConnectors;
					stopBit = numLayerConnectors;
					// NUMBER OF CONNECTORS FIT IN - BUT SOMETIMES CANNOT BE CENTERED ON slot
					if ((slot - halfConnectors < 0)) {
						// WE ARE TO CLOSE TO LEFT SIDE - ADD MORE INPUTS FROM RIGHT SIDE
						connectOffset = 0;
					}
					if ((slot + halfConnectors + 1 >= numLayerInputs)) { // +1 AS WE ARE TAKING CENTRAL NODE AS BELONGING TO RIGHT HALF
						// WE ARE TO CLOSE TO RIGHT SIDE - ADD MORE INPUTS FROM LEFT SIDE
						connectOffset = numLayerInputs - numLayerConnectors;
					}
				}

				for (int bit = 0; bit < stopBit; bit++) {
					int	bImplicitConnection = FALSE;
					int	bExplicitConnection = FALSE;
					bExplicitConnection = HasConnection(genome.gene(offsetFNC + slot), connect, slot, connectOffset, bit, &bImplicitConnection);
					if (bImplicitConnection || bExplicitConnection) {
						if (layer > 1) {
							int prevOffsetFNC = (layer - 2) * MAX_INTERNAL_LAYER_SIZE; 
							// ADD PREVIOUS NODE 	
							displayNodes[prevOffsetFNC + connectOffset + bit] = 1;
						}
					}		
				}
			}
		}
	}
	
	return status;
}

int CircuitGenome::HasConnection(unsigned long functionID, unsigned long connectionMask, int fncSlot, int connectionOffset, int bit, int* pbImplicitConnection) {
    int    status = FALSE;
    
    // DEFAULT: IF SIGNALIZED IN MASK, THAN ALLOW CONNECTION
    // SELECTED INSTRUCTION MAY CHANGE LATER
    if (connectionMask & (unsigned long) pGACirc->precompPow[bit]) status = TRUE;
    
    // IMPLICIT CONNECTION MAY EXIST
    switch (functionID) {
        case FNC_CONST: {
            // NO CONNECTION
            status = FALSE;
            break;
        }
        
        case FNC_SUBS: // no break
        case FNC_ADD:  // no break
        case FNC_MULT: // no break 
        case FNC_DIV: {
            // ALLOW *ALSO* CONNECTION FROM PREVIOUS LAYER ON SAME OFFSET
            if (fncSlot == connectionOffset + bit) {
                *pbImplicitConnection = TRUE;
            }
            break;
        }                    

        case FNC_NOP:  // no break
        case FNC_ROTL:  // no break
        case FNC_ROTR:  // no break
		case FNC_READX: // no break
        case FNC_BITSELECTOR: { 
            // ALLOW *ONLY* CONNECTION FROM PREVIOUS LAYER ON SAME OFFSET
            if (fncSlot == connectionOffset + bit) {
                *pbImplicitConnection = TRUE;
                
                // NO EXPLICIT CONNECTION POSSIBLE  
                status = FALSE;
            }
            else status = FALSE;
            break;
        }                    
    }
    
    return status;
}

int CircuitGenome::HasImplicitConnection(unsigned long functionID) {
    int    status = FALSE;
    
    switch (functionID) {
        case FNC_NOP: // no break
        case FNC_SUBS: // no break
        case FNC_ADD:  // no break
        case FNC_MULT: // no break 
        case FNC_DIV:  // no break
        case FNC_ROTL:  // no break
        case FNC_ROTR:  // no break
		case FNC_READX: // no break
        case FNC_BITSELECTOR: { 
            status = TRUE;
            break;
        }                    
    }
        
    return status;
}

int CircuitGenome::IsOperand(unsigned long functionID, unsigned long connectionMask, int fncSlot, int connectionOffset, int bit, string* pOperand) {
    int    status = FALSE;
    
    
    // DEFAULT: IF SIGNALIZED IN MASK, THAN ALLOW CONNECTION
    // SELECTED INSTRUCTION MAY CHANGE LATER
    if (connectionMask & (unsigned long) pGACirc->precompPow[bit]) status = TRUE;
    
    // IMPLICIT CONNECTION MAY EXIST
    switch (functionID) {
        case FNC_NOP:  {
            *pOperand = ""; status = FALSE; break;
        }
        case FNC_OR: {
            *pOperand = "|"; break;    
        }
        case FNC_AND: {
            *pOperand = "&"; break;    
        }
        case FNC_CONST: {
            if (fncSlot == connectionOffset + bit) {
                *pOperand = (connectionMask % UCHAR_MAX);
                status = TRUE;
            }
            else status = FALSE;
            break;
        }
		case FNC_READX: {
            if (fncSlot == connectionOffset + bit) {
                *pOperand = (connectionMask % UCHAR_MAX);
                status = TRUE;
            }
            else status = FALSE;
            break;
        }
        case FNC_XOR: {
            *pOperand = "^"; break;    
        }
        case FNC_NOR: {
            *pOperand = "| ~"; break;    
        }
        case FNC_NAND: {
            *pOperand = "& ~"; break;    
        }
        case FNC_ROTL: {
            if (fncSlot == connectionOffset + bit) {
                *pOperand = (connectionMask & 0x07);     
                status = TRUE;
            }
            else status = FALSE;
            break;
        }
        case FNC_ROTR: {
            if (fncSlot == connectionOffset + bit) {
                *pOperand = (connectionMask & 0x07);     
                status = TRUE;
            }
            else status = FALSE;
            break;
        }
        case FNC_BITSELECTOR: {
            if (fncSlot == connectionOffset + bit) {
                *pOperand = "&";     
                status = TRUE;
            }
            else status = FALSE;
            break;
        }
        case FNC_SUM: {
            *pOperand = "+"; break;
        }
        case FNC_SUBS: {
            *pOperand = "-"; break;     
        }
        case FNC_ADD: {
            *pOperand = "+"; break;    
        }  
        case FNC_MULT: {
            *pOperand = "*"; break;    
        }
        case FNC_DIV: {
            *pOperand = "/"; break;     
        }
        default: {
            *pOperand = "!!!";
            status = TRUE;
        }
    }
    
    return status;
}

int CircuitGenome::GetNeutralValue(unsigned long functionID, string* pOperand) {
    int    status = STAT_OK;
    
    switch (functionID) {
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

int CircuitGenome::ParseCircuit(string textCircuit, unsigned long* circuit, int* numLayers, int* intLayerSize, int* outLayerSize) {
    int           status = STAT_OK;
    unsigned int  pos = 0;
    unsigned int  pos2 = 0;
    
    // CHECK IF CIRCUIT IS GIVEN IN FORM OF GENOM ONLY OR TEXT REPRESENTATION
	if (textCircuit.find("[") != string::npos) {
		// TEXT REPRESENTATION
		*numLayers = 0;
        while ((pos2 = textCircuit.find(";", pos)) != string::npos) {
			string line = textCircuit.substr(pos, pos2 - pos + 1);
			//line.TrimLeft();line.TrimRight();
			TrimLeadingSpaces(line);
			TrimTrailingSpaces(line);
			if (line.find("[") != string::npos) {
				// AT LEAST ONE FNC ELEMENT PRESENT => NEW layer
				int offsetCON = (*numLayers * 2) * MAX_INTERNAL_LAYER_SIZE; 
				int offsetFNC = (*numLayers * 2 + 1) * MAX_INTERNAL_LAYER_SIZE; 
                unsigned int pos3 = 0;
                unsigned int pos4 = 0;
				int slot = 0;
				while ((pos4 = line.find("]", pos3)) != string::npos) {
					// PARSE ELEMENTS
					string elem = line.substr(pos3, pos4 - pos3 + 1);
					//elem.TrimLeft();elem.TrimRight();
					TrimLeadingSpaces(elem);
					TrimTrailingSpaces(elem);

					// CONNECTOR LAYER
					//DWORD conn1 = atol(elem);
					unsigned long conn = (unsigned long) StringToDouble(elem);
					//assert(conn1 == conn);
	                
					// FUNCTION
					unsigned long fnc = FNC_NOP;
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
	                
					circuit[offsetCON + slot] = conn;
					circuit[offsetFNC + slot] = fnc;
	                
					slot++; 
					pos3 = pos4 + 2;
				}
	            
				if (*numLayers == 0) *intLayerSize = slot; // NUMBER OF INTERNAL ELEMENTS IS SAME AS FOR FIRST LAYER
				*outLayerSize = slot; // WILL CONTAIN NUMBER OF ELEMENTS IN LAST LAYER
	            
				*numLayers = (*numLayers) + 1;
			}      
			pos = pos2 + 1;
		}
    }
    else {
		// TODO: OBTAIN CORRECT NUMBER OF LAYERS ETC - NOW TAKEN FROM USER INPUT

		// GENOM ONLY 
		int offset = 0;
        //while ((pos2 = textCircuit.find(" ", pos)) != string::npos) {
        while (offset < pGACirc->genomeSize) {
            pos2 = textCircuit.find(" ", pos);
            // manual change: commented line above substituted for 2 lines just above

			string value = textCircuit.substr(pos, pos2 - pos + 1);
			//value.TrimLeft();value.TrimRight();
			TrimLeadingSpaces(value);
			TrimTrailingSpaces(value);
			
			//DWORD temp = atol(value);
			circuit[offset] = (unsigned long) StringToDouble(value);
//			assert(temp == circuit[offset]);
			
			offset++;
			pos = pos2 + 1;
		}
    }
    
    return status;
}

int CircuitGenome::PrintCircuit(GAGenome &g, string filePath, unsigned char usePredictorMask[MAX_OUTPUTS], int bPruneCircuit) {
    int								status = STAT_OK;
    GA1DArrayGenome<unsigned long>&inputGenome = (GA1DArrayGenome<unsigned long>&) g;
	GA1DArrayGenome<unsigned long>  genome(MAX_GENOME_SIZE, Evaluator);
    string							message;
    string							value;
    string							value2;
    string							visualCirc = "";
    string							visualCircLabel = "";
    string							codeCirc = "";
	string 							actualSlotID; 
	string 							previousSlotID; 
	int								bCodeCircuit = TRUE;
	unsigned char					displayNodes[MAX_GENOME_SIZE];
	
	
    //
    // PRUNE CIRCUIT IF REQUIRED
    //
	if (pGACirc->allowPrunning && bPruneCircuit) {
        // PRUNE
        status = PruneCircuit(inputGenome, genome);    
        bCodeCircuit = TRUE;
        
		// COMPUTE NODES TO DISPLAY 
		memset(displayNodes, 0, sizeof(displayNodes));
		status = GetUsedNodes(genome, usePredictorMask, displayNodes);        
    }
    else {
        // JUST COPY
        for (int i = 0; i < genome.size(); i++) {
			genome.gene(i, inputGenome.gene(i)); 
		}
		// ALL NODES WILL BE DISPLAYED
		memset(displayNodes, 1, sizeof(displayNodes));
    }
    
    
    // VISUAL CIRC: INPUTS 
    visualCirc += "digraph EACircuit {\r\n\
rankdir=BT;\r\n\
edge [dir=none];\r\n\
size=\"6,6\";\r\n\
ordering=out;\r\n\
node [color=lightblue2, style=filled];\r\n";

    // CODE CIRCUIT: 
    if (bCodeCircuit) codeCirc += "static void circuit(unsigned char inputs[MAX_INPUTS], unsigned char outputs[MAX_OUTPUTS]) {\n";

    message += "\r\n";
    for (int layer = 1; layer < 2 * pGACirc->numLayers; layer = layer + 2) {
        int offsetCON = (layer-1) * MAX_INTERNAL_LAYER_SIZE; 
        int offsetFNC = (layer) * MAX_INTERNAL_LAYER_SIZE; 

        int numLayerInputs = 0;
        if (layer == 1) {
            // WHEN SECTORING IS USED THEN FIRST LAYER OBTAIN internalLayerSize INPUTS
            //if (pGACirc->bSectorInputData) numLayerInputs = pGACirc->internalLayerSize; 
            numLayerInputs = pGACirc->numInputs;
            
			// VISUAL CIRC: INPUTS 
			for (int i = 0; i < numLayerInputs; i++) {
				ostringstream os1;
				os1 << "IN_" << i;
				actualSlotID = os1.str();
				//value2.Format("\"%s\";\r\n", actualSlotID);
				ostringstream os2;
				os2 << "\"" << actualSlotID << "\";\r\n";
				value2 = os2.str();
				visualCirc += value2;
				
                if (bCodeCircuit) {
				    //value2.Format("    BYTE VAR_%s = inputs[%d];\r\n", actualSlotID, i);
					ostringstream os3;
					os3 << "    unsigned char VAR_" << actualSlotID << " = inputs[" << i << "];\n";
					value2 = os3.str();
				    codeCirc += value2;
				}
			}
    		codeCirc += "\n";
			
        }
        else numLayerInputs = pGACirc->internalLayerSize;

        int numFncs = pGACirc->internalLayerSize;
        // IF DISPLAYING THE LAST LAYER, THEN DISPLAY ONLY 'INTERNAL_LAYER_SIZE' FNC (OTHERS ARE UNUSED)
        if (layer == (2 * pGACirc->numLayers - 1)) numFncs = pGACirc->outputLayerSize;

		//
		// VISUAL CIRC: PUT ALL NODES FROM SAME LAYER INTO SAME RANK
		//
        value2 = "{ rank=same; ";
        for (int slot = 0; slot < numFncs; slot++) {
            // USE ONLY GENES THAT WERE NOT PRUNNED OUT
            //if (inputGenome.gene(offsetFNC + slot) == genome.gene(offsetFNC + slot)) {
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
            
			// SELECTOR LAYERS HAVE FULL INTERCONNECTION (CONNECTORS), OTHER HAVE SPECIFIED NUMBER 
			int	numLayerConnectors = pGACirc->numLayerConnectors;
			if (layer / 2 < pGACirc->numSelectorLayers) {
				numLayerConnectors = numLayerInputs;    
			}
			int	halfConnectors = (numLayerConnectors - 1) / 2;

			unsigned long effectiveCon = genome.gene(offsetCON + slot);
			//value2.Format("%.10u[%s]  ", effectiveCon, value);
			ostringstream os5;
			os5 <<  setfill('0') << setw(10) << effectiveCon << "[" << value << "]  ";
			value2 = os5.str();
			message += value2;
            
			// 
			// VISUAL CIRC: CREATE CONNECTION BETWEEN LAYERS
			//
			// USE ONLY GENES THAT WERE NOT PRUNNED OUT
//			if (inputGenome.gene(offsetFNC + slot) == genome.gene(offsetFNC + slot)) {
            if (displayNodes[offsetFNC + slot] == 1) {
				//actualSlotID.Format("%d_%d_%s", layer / 2 + 1, slot, value);
				ostringstream os6;
				os6 << (layer / 2 + 1) << "_" << slot << "_" << value;
				actualSlotID = os6.str();

				int connectOffset = 0;
				int stopBit = 0;
				if (numLayerConnectors > numLayerInputs) {
					// NUMBER OF CONNECTORS IS HIGHER THAN NUMBER OF FUNCTIONS IN LAYER - CUT ON BOTH SIDES			
					connectOffset = 0;
					stopBit = numLayerInputs;
				}
				else {
					connectOffset = slot - halfConnectors;
					stopBit = numLayerConnectors;
					// NUMBER OF CONNECTORS FIT IN - BUT SOMETIMES CANNOT BE CENTERED ON slot
					if ((slot - halfConnectors < 0)) {
						// WE ARE TO CLOSE TO LEFT SIDE - ADD MORE INPUTS FROM RIGHT SIDE
						connectOffset = 0;
					}
					if ((slot + halfConnectors + 1 >= numLayerInputs)) { // +1 AS WE ARE TAKING CENTRAL NODE AS BELONGING TO RIGHT HALF
						// WE ARE TO CLOSE TO RIGHT SIDE - ADD MORE INPUTS FROM LEFT SIDE
						connectOffset = numLayerInputs - numLayerConnectors;
					}
				}
			 
				int    bFirstArgument = TRUE;
				int    bAtLeastOneConnection = FALSE;
			    
				// CODE CIRCUIT: 
				if (bCodeCircuit) {
					switch (genome.gene(offsetFNC + slot)) {
						case FNC_CONST: {
							//value2.Format("    BYTE VAR_%s = %u", actualSlotID, effectiveCon % UCHAR_MAX);
							ostringstream os7;
							os7 << "    unsigned char VAR_" << actualSlotID << " = " << (effectiveCon % UCHAR_MAX);
							value2 = os7.str();
							codeCirc += value2;
							bFirstArgument = FALSE;
							bAtLeastOneConnection = TRUE;
							break;
						}
						case FNC_NOR: {
							//value2.Format("    unsigned char VAR_%s = 0 | ~", actualSlotID);
							ostringstream os8;
							os8 << "    unsigned char VAR_" << actualSlotID << " = 0 | ~"; 
							value2 = os8.str();
							codeCirc += value2;
							bFirstArgument = FALSE;
							bAtLeastOneConnection = TRUE;
							break;
						}
						case FNC_NAND: {
							//value2.Format("    unsigned char VAR_%s = 0xff & ~ ", actualSlotID);
							ostringstream os9;
							os9 << "    unsigned char VAR_" << actualSlotID << " = 0xff & ~ "; 
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
					if (HasImplicitConnection(genome.gene(offsetFNC + slot))) {
						if (layer > 1) {
							int prevOffsetFNC = (layer - 2) * MAX_INTERNAL_LAYER_SIZE; 
							int prevOffsetCON = (layer - 3) * MAX_INTERNAL_LAYER_SIZE; 
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
						os12 << "    unsigned char VAR_" << actualSlotID << " =";
						value2 = os12.str();
						codeCirc += value2;

						ostringstream os13;
	                    					    
						switch (genome.gene(offsetFNC + slot)) {
							case FNC_NOP: os13 << " VAR_" << previousSlotID; value2 = os13.str(); break; 
							case FNC_SUBS: os13 << " VAR_" << previousSlotID << " - "; value2 = os13.str(); break; 
							case FNC_ADD: os13 << " VAR_" << previousSlotID << " + "; value2 = os13.str(); break; 
							case FNC_MULT: os13 << " VAR_" << previousSlotID << " * "; value2 = os13.str(); break; 
							case FNC_DIV: os13 << " VAR_" << previousSlotID << " / "; value2 = os13.str(); break; 
							case FNC_ROTL: os13 << " VAR_" << previousSlotID << " << " << (effectiveCon & 0x07); value2 = os13.str(); break; 
							case FNC_ROTR: os13 << " VAR_" << previousSlotID << " >> " << (effectiveCon & 0x07); value2 = os13.str(); break; 
							case FNC_BITSELECTOR: os13 << " VAR_" << previousSlotID + " & " << (effectiveCon & 0x07); value2 = os13.str(); break; 
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
							default: value2 = "!!!";
						}
						codeCirc += value2;

						bFirstArgument = FALSE;
						bAtLeastOneConnection = FALSE;
					}
				}
			    
				for (int bit = 0; bit < stopBit; bit++) {
					// IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE INPUT
					int    bImplicitConnection = FALSE;
					int    bExplicitConnection = FALSE;
	                
					bExplicitConnection = HasConnection(genome.gene(offsetFNC + slot), effectiveCon, slot, connectOffset, bit, &bImplicitConnection);
					if (bImplicitConnection || bExplicitConnection) {
						bAtLeastOneConnection = TRUE;
	                    
						if (layer > 1) {
							int prevOffsetFNC = (layer - 2) * MAX_INTERNAL_LAYER_SIZE; 
							int prevOffsetCON = (layer - 3) * MAX_INTERNAL_LAYER_SIZE; 
							GetFunctionLabel(genome.gene(prevOffsetFNC + connectOffset + bit), genome.gene(prevOffsetCON + connectOffset + bit), &value);
							//previousSlotID.Format("%d_%d_%s", layer / 2, connectOffset + bit, value);
							ostringstream os21;
							os21 << (layer / 2) << "_" << (connectOffset + bit) << "_" << value;
							previousSlotID = os21.str();
						}
						else {
							//previousSlotID.Format("IN_%d", connectOffset + bit);
							ostringstream os22;
							os22 << "IN_" << (connectOffset + bit);
							previousSlotID = os22.str();
						}
						if (bExplicitConnection) {
							//value2.Format("\"%s\" -> \"%s\";\r\n", actualSlotID, previousSlotID);
							ostringstream os23;
							os23 << "\"" << actualSlotID << "\" -> \"" << previousSlotID << "\";\r\n";
							value2 = os23.str();
							visualCirc += value2;
						}
						if (bImplicitConnection) {
							// USE ONLY GENES THAT WERE NOT PRUNNED OUT
							//value2.Format("\"%s\" -> \"%s\" [color=red];\r\n", actualSlotID, previousSlotID);
							ostringstream os24;
							os24 << "\"" << actualSlotID << "\" -> \"" << previousSlotID << "\" [color=red];\r\n";
							value2 = os24.str();
							visualCirc += value2;
						}
	                    
						if (bCodeCircuit) {
							string operand;
							if (bFirstArgument) {
								//value2.Format("    unsigned char VAR_%s =", actualSlotID);
								ostringstream os25;
								os25 << "    unsigned char VAR_" << actualSlotID << " =";
								value2 = os25.str();
								codeCirc += value2;
								bFirstArgument = FALSE;
							}
							if (bExplicitConnection) {
								if (IsOperand(genome.gene(offsetFNC + slot), effectiveCon, slot, connectOffset, bit, &operand)) {
									if (genome.gene(offsetFNC + slot) == FNC_DIV) {
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
						os28 << "    unsigned char VAR_" << actualSlotID << " = 0";
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
    
    // VISUAL CIRC: CONNECT OUTPUT LAYER
    for (int i = 0; i < pGACirc->outputLayerSize; i++) {
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
		
		if (!bPruneCircuit || usePredictorMask == NULL || usePredictorMask[i] == 1) {
		    //actualSlotID.Format("%d_OUT", i);
			ostringstream os29;
			os29 << i << "_OUT";
			actualSlotID = os29.str();
		    int prevOffsetFNC = (2 * pGACirc->numLayers - 1) * MAX_INTERNAL_LAYER_SIZE; 
		    int prevOffsetCON = (2 * pGACirc->numLayers - 2) * MAX_INTERNAL_LAYER_SIZE; 
		    string		value;
		    GetFunctionLabel(genome.gene(prevOffsetFNC + i), genome.gene(prevOffsetCON + i), &value);
		    //previousSlotID.Format("%d_%d_%s", pGACirc->numLayers, i, value);
			ostringstream os30;
			os30 << (pGACirc->numLayers) << "_" << i << "_" << value;
			previousSlotID = os30.str();
		    //value2.Format("\"%s\" -> \"%s\";\r\n", actualSlotID, previousSlotID);
			ostringstream os31;
			os31 << "\"" << actualSlotID << "\" -> \"" << previousSlotID << "\";\r\n";
			value2 = os31.str();
		    visualCirc += value2;
		    
            if (bCodeCircuit) {
			    //value2.Format("    outputs[%d] = VAR_%s;\r\n", i, previousSlotID);
				ostringstream os32;
				os32 << "    outputs[" << i << "] = VAR_" << previousSlotID << ";\n";
				value2 = os32.str();
			    codeCirc += value2;
            }
		}
    }
    
    if (bCodeCircuit) codeCirc += "\n}";
    
    message += "\r\n";

    visualCirc += "}";

	//
	//	STORE IN SPECIFIED FILES
	//
    if (filePath == "") filePath = FILE_BEST_CIRCUIT;

	fstream	file;
	string	newFilePath;

	/*int num=0;
	while (true) {

		std::string s;
		std::stringstream out;
		out << ++num;
		s = out.str();

		ifstream ifile(s + filePath + ".txt");
		if (!ifile) break;
	} 

	std::string s;
	std::stringstream out;
	out << num;
	s = out.str();
	filePath = s + filePath;
	*/

	// TEXT CIRCUIT
	//newFilePath.Format("%s.txt", filePath);
	newFilePath = filePath + ".txt";
	file.open(newFilePath.c_str(), fstream::in | fstream::out | fstream::ate | fstream::trunc);
	if (file.is_open()) {
		file << message;
		file.close();
	}

	// BINARY GENOM
	//newFilePath.Format("%s.bin", filePath);
	newFilePath = filePath + ".bin";
	message = "";
	file.open(newFilePath.c_str(), fstream::in | fstream::out | fstream::ate | fstream::trunc);
	if (file.is_open()) {
	    for (int i = 0; i < genome.length(); i++) {
			//file << (genome.gene(i)) << " ";
			ostringstream os33;
			os33 << genome.gene(i) << " ";
			value = os33.str();
			file << value;
			message += value;
	    }
		file.close();
	}
    message += "\r\n";

	// GRAPHVIZ FILE
	//newFilePath.Format("%s.dot", filePath);
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


	// C FILE
	if (bCodeCircuit) {
	    //newFilePath.Format("%s.c", filePath);
		newFilePath = filePath + ".c";
		file.open(newFilePath.c_str(), fstream::in | fstream::out | fstream::ate | fstream::trunc);
		if (file.is_open()) {
		    file << codeCirc;
		    file.close();
	    }
    }


    /*
	//
	//	STORE IN DEFAULT FILE
	//
    filePath = FILE_BEST_CIRCUIT;
	
	// TEXT CIRCUIT
	//newFilePath.Format("%s.txt", filePath);
	newFilePath = filePath + ".txt";
	file.open(newFilePath.c_str(), fstream::in | fstream::out | fstream::ate | fstream::trunc);
	if (file.is_open()) {
		file << message;	
		file.close();
	}

	// BINARY GENOM
	//newFilePath.Format("%s.bin", filePath);
	newFilePath = filePath + ".bin";
	file.open(newFilePath.c_str(), fstream::in | fstream::out | fstream::ate | fstream::trunc);
	if (file.is_open()) {
	    for (int i = 0; i < genome.length(); i++) {
			//value.Format("%u ", genome.gene(i));
			ostringstream os33;
			os33 << genome.gene(i) << " ";
			value = os33.str();
			file << value;
	    }
		file.close();
	}

	// GRAPHVIZ FILE
	//newFilePath.Format("%s.dot", filePath);
	newFilePath = filePath + ".dot";
	file.open(newFilePath.c_str(), fstream::in | fstream::out | fstream::ate | fstream::trunc);
	if (file.is_open()) {
		file << visualCirc;
		file.close();
	}
    */
    
    return status;
}

int CircuitGenome::ExecuteCircuit(GA1DArrayGenome<unsigned long>* pGenome, unsigned char inputs[MAX_INPUTS], unsigned char outputs[MAX_OUTPUTS]) {  
    int     status = STAT_OK;
    unsigned char*   inputsBegin = inputs;
    int     numSectors = 1;
    int     sectorLength = pGACirc->numInputs;
    unsigned char    localInputs[MAX_INPUTS];
    
    memset(outputs, 0, MAX_OUTPUTS);
    
    // ALL IN ONE RUN
    numSectors = 1;
    sectorLength = pGACirc->numInputs;
    
    for (int sector = 0; sector < numSectors; sector++) { 
        // PREPARE INPUTS FOR ACTUAL RUN OF CIRCUIT
        if (numSectors == 1) {
            // ALL INPUT DATA AT ONCE
            memcpy(localInputs, inputs, MAX_INPUTS);
        }
        else {
            // USE STATE (OUTPUT) AS FIRST PART OF INPUT
            memcpy(localInputs, outputs, pGACirc->outputLayerSize);
            // ADD FRESH INPUT DATA
            memcpy(localInputs + pGACirc->outputLayerSize, inputs + sector * sectorLength, sectorLength);
        }
        
        // EVALUATE CIRCUIT
        for (int layer = 1; layer < 2 * pGACirc->numLayers; layer = layer + 2) {
            int offsetCON = (layer-1) * MAX_INTERNAL_LAYER_SIZE; 
            int offsetFNC = (layer) * MAX_INTERNAL_LAYER_SIZE; 
            memset(outputs, 0, MAX_INTERNAL_LAYER_SIZE);

            // actual number of inputs for this layer. For first layer equal to pCircuit->numInputs, for next layers equal to number of function in intermediate layer pCircuit->internalLayerSize
            int numLayerInputs = 0;
            if (layer == 1) {
                // WHEN SECTORING IS USED THEN FIRST LAYER OBTAIN internalLayerSize INPUTS
                //if (pCircuit->bSectorInputData) numLayerInputs = pCircuit->internalLayerSize; 
                numLayerInputs = pGACirc->numInputs;
            }
            else numLayerInputs = pGACirc->internalLayerSize;

            // actual number of functions in layer - different for the last "output" layer
            int numFncInLayer = (layer == (2 * pGACirc->numLayers - 1)) ? pGACirc->outputLayerSize : pGACirc->internalLayerSize; 

            // SELECTOR LAYERS HAVE FULL INTERCONNECTION (CONNECTORS), OTHER HAVE SPECIFIED NUMBER 
    	    int	numLayerConnectors = pGACirc->numLayerConnectors;
            if (layer / 2 < pGACirc->numSelectorLayers) {
                numLayerConnectors = numLayerInputs;    
            }
    	    int	halfConnectors = (numLayerConnectors - 1) / 2;

            for (int slot = 0; slot < numFncInLayer; slot++) {
                unsigned char	result = 0;
                unsigned long   connect = 0;
                int     connectOffset = 0;
                int     stopBit = 0;
                
                // COMPUTE RANGE OF INPUTS FOR PARTICULAR slot FUNCTION
                connect = pGenome->gene(offsetCON + slot);
    			
			    if (numLayerConnectors > numLayerInputs) {
				    // NUMBER OF CONNECTORS IS HIGHER THAN NUMBER OF FUNCTIONS IN LAYER - CUT ON BOTH SIDES			
				    connectOffset = 0;
				    stopBit = numLayerInputs;
			    }
			    else {
				    connectOffset = slot - halfConnectors;
				    stopBit = numLayerConnectors;
				    // NUMBER OF CONNECTORS FIT IN - BUT SOMETIMES CANNOT BE CENTERED ON slot
				    if ((slot - halfConnectors < 0)) {
					    // WE ARE TO CLOSE TO LEFT SIDE - ADD MORE INPUTS FROM RIGHT SIDE
					    connectOffset = 0;
				    }
				    if ((slot + halfConnectors + 1 >= numLayerInputs)) { // +1 AS WE ARE TAKING CENTRAL NODE AS BELONGING TO RIGHT HALF
					    // WE ARE TO CLOSE TO RIGHT SIDE - ADD MORE INPUTS FROM LEFT SIDE
					    connectOffset = numLayerInputs - numLayerConnectors;
				    }
			    }
				//for (int y = 0; y < 16; y++) cout << (int)localInputs[y] << ":";
				//cout << endl;
                switch (pGenome->gene(offsetFNC + slot)) {
                    case FNC_NOP: {
                        // DO NOTHING, JUST PASS VALUE FROM PREVIOUS LAYER
                        result = localInputs[slot];
                        break;
                    }
                    case FNC_OR: {
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE INPUT
                            if (connect & (unsigned long) pGACirc->precompPow[bit]) {
                                result |= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_AND: {
                        result = ~0; // ASSIGN ALL ONES AS STARTING VALUE TO PERFORM result &= LATER
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION, THEN TAKE INPUT
                            if (connect & (unsigned long) pGACirc->precompPow[bit]) {
                                result &= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_CONST: {
                        // SEND VALUE FROM CONNECTION LAYER DIRECTLY TO OUTPUT
                        result = pGenome->gene(offsetCON + slot) % UCHAR_MAX;
                        break;
                    }
                    case FNC_XOR: {
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE INPUT
                            if (connect & (unsigned long) pGACirc->precompPow[bit]) {
                                result ^= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_NOR: {
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE NEGATION OF INPUT
                            if (connect & (unsigned long) pGACirc->precompPow[bit]) {
                                result |= ~(localInputs[connectOffset + bit]);
                            }
                        }
                        break;
                    }
                    case FNC_NAND: {
                        result = ~0; // ASSIGN ALL ONES AS STARTING VALUE TO PERFORM result &= LATER
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE NEGATION OF INPUT
                            if (connect & (unsigned long) pGACirc->precompPow[bit]) {
                                result &= ~(localInputs[connectOffset + bit]);
                            }
                        }
                        break;
					}
                    case FNC_ROTL: {
                        // TAKE INPUT FROM THE FUNCTION FROM PREVIOUS LAYER ON THE SAME POSITION AS THIS
                        // MAXIMIM SHIFT IS 7
                        result = localInputs[slot] << (pGenome->gene(offsetCON + slot) & 0x07);
                        break;
                    }
                    case FNC_ROTR: {
                        // TAKE INPUT FROM THE FUNCTION FROM PREVIOUS LAYER ON THE SAME POSITION AS THIS
                        // MAXIMIM SHIFT IS 7
                        result = localInputs[slot] >> (pGenome->gene(offsetCON + slot) & 0x07);
                        break;
                    }
                    case FNC_BITSELECTOR: {
                        // BIT SELECTOR
                        // TAKE INPUT FROM THE FUNCTION FROM PREVIOUS LAYER ON THE SAME POSITION AS THIS
                        // OUTPUT ONLY BITS ON POSITION GIVEN BY THE MASK FROM CONNECTION LAYER
                        result = localInputs[slot] & (pGenome->gene(offsetCON + slot) & 0x07);
                        break;
                    }
                    case FNC_SUM: {
                        // SUM ALL INPUTS
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE NEGATION OF INPUT
                            if (connect & (unsigned long) pGACirc->precompPow[bit]) {
                                result += localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_SUBS: {
                        // TAKE INPUT FROM THE FUNCTION FROM PREVIOUS LAYER ON THE SAME POSITION AS THIS ONE
                        // AND SUBSTRACT ALL VALUES FROM CONNECTED INPUTS
                        result = localInputs[slot];
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE NEGATION OF INPUT
                            if (connect & (unsigned long) pGACirc->precompPow[bit]) {
                                result -= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_ADD: {
                        // TAKE INPUT FROM THE FUNCTION FROM PREVIOUS LAYER ON THE SAME POSITION AS THIS ONE
                        // AND ADD ALL VALUES FROM CONNECTED INPUTS
                        result = localInputs[slot];
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE NEGATION OF INPUT
                            if (connect & (unsigned long) pGACirc->precompPow[bit]) {
                                result += localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_MULT: {
                        // TAKE INPUT FROM THE FUNCTION FROM PREVIOUS LAYER ON THE SAME POSITION AS THIS ONE
                        // AND MULTIPLY IT BY ALL VALUES FROM CONNECTED INPUTS
                        result = localInputs[slot];
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE NEGATION OF INPUT
                            if (connect & (unsigned long) pGACirc->precompPow[bit]) {
                                result *= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_DIV: {
                        // TAKE INPUT FROM THE FUNCTION FROM PREVIOUS LAYER ON THE SAME POSITION AS THIS ONE
                        // AND DIVIDE IT BY ALL VALUES FROM CONNECTED INPUTS
                        result = localInputs[slot];
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE NEGATION OF INPUT
                            if (connect & (unsigned long) pGACirc->precompPow[bit]) {
                                if (localInputs[connectOffset + bit] != 0) {
                                    result /= localInputs[connectOffset + bit];
                                }
                            }
                        }
                        break;
                    }
					case FNC_READX: {
						// READ x-th byte
						result = localInputs[slot];
						for (int bit = 0; bit < stopBit; bit++) {
							if (connect & (unsigned long) pGACirc->precompPow[bit]) {
								result += localInputs[connectOffset + bit];
							}
						}
						result = inputs[result % pGACirc->numInputs];
                        break;
					}
                    default: {
                        assert(FALSE);
                        break;
                    }
				}
                
                outputs[slot] = result;
            }
            // PREPARE INPUTS FOR NEXT LAYER FROM OUTPUTS
            memcpy(localInputs, outputs, pGACirc->internalLayerSize);
			//cout << endl;
        }
    }
    
    return status;
}

int CircuitGenome::writeGenome(const GA1DArrayGenome<unsigned long>& genome, string& textCircuit) {
    int status = STAT_OK;

    ostringstream textCicruitStream;
    for (int i = 0; i < genome.length(); i++) {
        textCicruitStream << genome.gene(i) << " ";
    }
    textCircuit = textCicruitStream.str();

    return status;
}

int CircuitGenome::readGenome(GA1DArrayGenome<unsigned long>& genome, string& textCircuit) {
    if (genome.size() != pGACirc->genomeSize) {
        mainLogger.out() << "error: Incorrect genome size. Cannot read genome." << endl;
        return STAT_INCOMPATIBLE_PARAMETER;
    }
    unsigned long* circuit = new unsigned long[pGACirc->genomeSize];

    memset(circuit, 0, sizeof(circuit));
    CircuitGenome::ParseCircuit(textCircuit, circuit, &(pGACirc->numLayers), &(pGACirc->internalLayerSize), &(pGACirc->outputLayerSize));

    // TODO: always keep real genome size, not constant from config file
    // ACTUAL SIZE OF GENOM
    // pGACirc->genomeSize = (pGACirc->numLayers * 2) * MAX_INTERNAL_LAYER_SIZE;

    for (int i = 0; i < pGACirc->genomeSize; i++) genome.gene(i, circuit[i]);

    delete[] circuit;
    // TODO: add circiut parsing status
    return STAT_OK;
}

int CircuitGenome::writePopulation(const GAPopulation& population, ostream& out) {
    int status = STAT_OK;
    string textCircuit;
    for (int i = 0; i < population.size(); i++) {
        // note: it is not necessary to call individual i in SCALED order
        //       however then the population files differ in order ('diff' cannot be used to fing bugs)
        GA1DArrayGenome<unsigned long>* pGenome = (GA1DArrayGenome<unsigned long>*) &(population.individual(i,GAPopulation::SCALED));
        status = CircuitGenome::writeGenome(*pGenome ,textCircuit);
        if (status != STAT_OK) {
            return status;
        }
        out << textCircuit << endl;
    }
    return status;
}

int CircuitGenome::readPopulation(GAPopulation& population, istream& in) {
    int populationSize = 0;
    int fileGenomeSize = 0;
    string line;
    getline(in,line);
    istringstream(line) >> populationSize;
    getline(in,line);
    istringstream(line) >> fileGenomeSize;
    if (fileGenomeSize != pGACirc->genomeSize) {
        mainLogger.out() << "error when loading population: Genomes size incorrect." << endl;
        return STAT_INCOMPATIBLE_PARAMETER;
    }

    GA1DArrayGenome<unsigned long> genome(pGACirc->genomeSize, CircuitGenome::Evaluator);
    // INIT GENOME STRUCTURES
    genome.initializer(CircuitGenome::Initializer);
    genome.mutator(CircuitGenome::Mutator);
    genome.crossover(CircuitGenome::Crossover);
    // LOAD genomes
    string textCircuit;
    for (int i = 0; i < populationSize; i++) {
        getline(in, textCircuit);
        CircuitGenome::readGenome(genome,textCircuit);
        population.add(genome);
    }

    return STAT_OK;
}
