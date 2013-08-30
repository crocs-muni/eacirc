#include <iomanip>
#include "CircuitGenome.h"
#include "CommonFnc.h"
#include "evaluators/IEvaluator.h"
#include "XMLProcessor.h"
#include "GAPopulation.h"
#include "generators/IRndGen.h"
#include "circuit/GACallbacks.h"
#include "circuit/CircuitIO.h"

int CircuitGenome::GetFunctionLabel(GENOME_ITEM_TYPE functionID, GENOME_ITEM_TYPE connections, string* pLabel) {
    int		status = STAT_OK;
    switch (nodeGetFunction(functionID)) {
        case FNC_NOP: *pLabel = "NOP"; break;
        case FNC_OR: *pLabel = "OR_"; break;
        case FNC_AND: *pLabel = "AND"; break;
        case FNC_CONST: {
			std::stringstream out;
            out << (nodeGetArgument1(functionID)  & 0xff);
			*pLabel = "CONST_" + out.str();
            break;
        }
		case FNC_READX: *pLabel = "RDX"; break;
        case FNC_XOR: *pLabel = "XOR"; break;
        case FNC_NOR: *pLabel = "NOR"; break;
        case FNC_NAND: *pLabel = "NAN"; break;
        case FNC_ROTL: {
			std::stringstream out;
            unsigned char tmp = nodeGetArgument1(functionID);
            out << (nodeGetArgument1(functionID) & 0x07);
            *pLabel = "ROL_" + out.str(); 
            break;
        }
        case FNC_ROTR: {
			std::stringstream out;
            out << (nodeGetArgument1(functionID) & 0x07);
            *pLabel = "ROR_" + out.str(); 
            break;
        }
        case FNC_BITSELECTOR: {
			std::stringstream out;
            out << (nodeGetArgument1(functionID) & 0x07);
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
    GA1DArrayGenome<GENOME_ITEM_TYPE>  &genome = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) g;
    GA1DArrayGenome<GENOME_ITEM_TYPE>  &prunnedGenome = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) prunnedG;
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
        
        float origFit = GACallbacks::evaluator(prunnedGenome);
        
        int prunneRepeat = 0; 
        bChangeDetected = TRUE;
        while (bChangeDetected && prunneRepeat < 10) {
            bChangeDetected = FALSE;
            prunneRepeat++;
            
            // DISABLE GENES STARTING FROM END 
            for (int i = prunnedGenome.size() - 1; i >= 0; i--) {
                GENOME_ITEM_TYPE   origValue = prunnedGenome.gene(i);
                
                if (origValue != 0) {
                    // PRUNE FNC AND CONNECTION LAYER DIFFERENTLY
                    if (((i / pGlobals->settings->circuit.genomeWidth) % 2) == 1) {
                        // FNCs LAYER - TRY TO SET AS NOP INSTRUCTION WITH NO CONNECTORS
                        prunnedGenome.gene(i, FNC_NOP);
                        
                        assert(nodeGetFunction(origValue) <= FNC_MAX);
                        
                        float newFit = GACallbacks::evaluator(prunnedGenome);
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
                        GENOME_ITEM_TYPE   tempOrigValue = origValue;  // WILL HOLD MASK OF IMPORTANT CONNECTIONS
                        // CONNECTION LAYER - TRY TO REMOVE CONNECTIONS GRADUALLY
                        for (int conn = 0; conn < MAX_LAYER_SIZE; conn++) {
                            GENOME_ITEM_TYPE   newValue = tempOrigValue & (~pGlobals->precompPow[conn]);
                            
                            if (newValue != tempOrigValue) {
                                prunnedGenome.gene(i, newValue);
                                
                                float newFit = GACallbacks::evaluator(prunnedGenome);
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
    GA1DArrayGenome<GENOME_ITEM_TYPE>  &genome = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) g;
    GA1DArrayGenome<GENOME_ITEM_TYPE>  &prunnedGenome = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) prunnedG;
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
        
        float origFit = GACallbacks::evaluator(prunnedGenome);
        
        int prunneRepeat = 0; 
        bChangeDetected = TRUE;
        while (bChangeDetected && prunneRepeat < 10) {
            bChangeDetected = FALSE;
            prunneRepeat++;
            
			// DISABLE GENES STARTING FROM END 
			for (int layer = 1; layer < 2 * pGlobals->settings->circuit.numLayers; layer = layer + 2) {
                int offsetCON = (layer-1) * pGlobals->settings->circuit.genomeWidth;
                int offsetFNC = (layer) * pGlobals->settings->circuit.genomeWidth;

				// actual number of functions in layer - different for the last "output" layer
                int numFncInLayer = (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) ? (pGlobals->settings->circuit.sizeOutputLayer) : pGlobals->settings->circuit.sizeLayer;

				for (int slot = 0; slot < numFncInLayer; slot++) {
                    GENOME_ITEM_TYPE   origValueFnc = genome.gene(offsetFNC + slot);
                    GENOME_ITEM_TYPE   origValueCon = genome.gene(offsetCON + slot);

					// TRY TO SET AS NOP INSTRUCTION WITH NO CONNECTORS
					prunnedGenome.gene(offsetFNC + slot, FNC_NOP);	// NOP
					prunnedGenome.gene(offsetCON + slot, 0);		// NO CONNECTORS

                    float newFit = GACallbacks::evaluator(prunnedGenome);
                    if (origFit > newFit) {
                        // SOME PART OF THE GENE WAS IMPORTANT, SET BACK 
						prunnedGenome.gene(offsetFNC + slot, origValueFnc);	
						prunnedGenome.gene(offsetCON + slot, origValueCon);		

                        // TRY TO REMOVE CONNECTIONS GRADUALLY
                        GENOME_ITEM_TYPE   prunnedConnectors = origValueCon;
                        for (int conn = 0; conn < MAX_LAYER_SIZE; conn++) {
                            GENOME_ITEM_TYPE   newConValue = prunnedConnectors & (~pGlobals->precompPow[conn]);
                            
                            if (newConValue != prunnedConnectors) {
                                prunnedGenome.gene(offsetCON + slot, newConValue);
                                
                                float newFit = GACallbacks::evaluator(prunnedGenome);
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
    GA1DArrayGenome<GENOME_ITEM_TYPE>  &genome = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) g;
	
	//
	// BUILD SET OF USED NODES FROM OUTPUT TO INPUT
	//
	
	// ADD OUTPUT NODES
    // VISUAL CIRC: CONNECT OUTPUT LAYER
    int offsetFNC = (2 * pGlobals->settings->circuit.numLayers - 1) * pGlobals->settings->circuit.genomeWidth;
    for (int i = 0; i < pGlobals->settings->circuit.sizeOutputLayer; i++) {
		if (usePredictorMask == NULL || usePredictorMask[i] == 1) {
			// ADD THIS ONE TO LIST OF USED NODES 
			displayNodes[offsetFNC + i] = 1;	
		}
    }
	
	// PROCESS ALL LAYERS FROM BACK
    for (int layer = 2 * pGlobals->settings->circuit.numLayers - 1; layer > 0; layer = layer - 2) {
        int offsetCON = (layer-1) * pGlobals->settings->circuit.genomeWidth;
        int offsetFNC = (layer) * pGlobals->settings->circuit.genomeWidth;

        // actual number of inputs for this layer. For first layer equal to pCircuit->numInputs, for next layers equal to number of function in intermediate layer pCircuit->internalLayerSize
        int numLayerInputs = 0;
        if (layer == 1) {
            // WHEN SECTORING IS USED THEN FIRST LAYER OBTAIN internalLayerSize INPUTS
            //if (pGACirc->bSectorInputData) numLayerInputs = pGACirc->internalLayerSize; 
            numLayerInputs = pGlobals->settings->circuit.sizeInput;
        }
        else numLayerInputs = pGlobals->settings->circuit.sizeLayer;

        // actual number of functions in layer - different for the last "output" layer
        int numFncInLayer = (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) ? pGlobals->settings->circuit.sizeOutputLayer : pGlobals->settings->circuit.sizeLayer;

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
            GENOME_ITEM_TYPE   connect = 0;
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
                            int prevOffsetFNC = (layer - 2) * pGlobals->settings->circuit.genomeWidth;
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

int CircuitGenome::FilterEffectiveConnections(GENOME_ITEM_TYPE functionID, GENOME_ITEM_TYPE connectionMask, int numLayerConnectors, GENOME_ITEM_TYPE* pEffectiveConnectionMask) {
	int	status = STAT_OK;

	*pEffectiveConnectionMask = 0;

    switch (nodeGetFunction(functionID)) {
		// FUNCTIONS WITH ONLY ONE CONNECTOR
		case FNC_NOP:  // no break
        case FNC_ROTL: // no break
        case FNC_ROTR: // no break
        case FNC_BITSELECTOR: {
			// Include only first connector bit
			for (int i = 0; i < numLayerConnectors; i++) {
                if (connectionMask & (GENOME_ITEM_TYPE) pGlobals->precompPow[i])  {
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
                if (connectionMask & (GENOME_ITEM_TYPE) pGlobals->precompPow[i])  {
					*pEffectiveConnectionMask +=  pGlobals->precompPow[i];
				}
			}
			break;
        }
    }

	return status;
}

int CircuitGenome::HasConnection(GENOME_ITEM_TYPE functionID, GENOME_ITEM_TYPE connectionMask, int fncSlot, int connectionOffset, int bit) {
    int    bHasConnection = FALSE; // default is NO
    
    // DEFAULT: IF SIGNALIZED IN CONNECTOR MASK, THAN ALLOW CONNECTION
    // SOME INSTRUCTION MAY CHANGE LATER
    if (connectionMask & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) bHasConnection = TRUE;
	else bHasConnection = FALSE;
	
    switch (nodeGetFunction(functionID)) {
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
                if (connectionMask & (GENOME_ITEM_TYPE) pGlobals->precompPow[i]) bHasConnection = FALSE;
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

int CircuitGenome::IsOperand(GENOME_ITEM_TYPE functionID, GENOME_ITEM_TYPE connectionMask, int fncSlot, int connectionOffset, int bit, string* pOperand) {
    int    bHasConnection = HasConnection(functionID, connectionMask, fncSlot, connectionOffset, bit);
    
    switch (nodeGetFunction(functionID)) {
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
                out << nodeGetArgument1(functionID);
				*pOperand = out.str();
			}
			break;
        }

        case FNC_ROTL: // no break
        case FNC_ROTR: {
			if (bHasConnection) {
				std::stringstream out;
                unsigned char tmp = nodeGetArgument1(functionID);
                if (nodeGetFunction(functionID) == FNC_ROTL) out << "<< " << (nodeGetArgument1(functionID) & 0x07);
                if (nodeGetFunction(functionID) == FNC_ROTR) out << ">> " << (nodeGetArgument1(functionID) & 0x07);
				*pOperand = out.str();
			}
			break;
        }
        case FNC_BITSELECTOR: {
			if (bHasConnection) {
				std::stringstream out;
                out << " & " << (nodeGetArgument1(functionID) & 0xff);
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

int CircuitGenome::GetNeutralValue(GENOME_ITEM_TYPE functionID, string* pOperand) {
    int    status = STAT_OK;
    
    switch (nodeGetFunction(functionID)) {
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

/*
int CircuitGenome::readGenomeFromBinary(string textCircuit, GA1DArrayGenome<GENOME_ITEM_TYPE>* genome) {
    istringstream circuitStream(textCircuit);
    GENOME_ITEM_TYPE gene;
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
*/

// TODO/TBD change according to printcircuit
int CircuitGenome::readGenomeFromText(string textCircuit, GA1DArrayGenome<GENOME_ITEM_TYPE>* genome) {
    GENOME_ITEM_TYPE* circuit = new GENOME_ITEM_TYPE[pGlobals->settings->circuit.genomeSize];

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
            int offsetCON = (local_numLayers * 2) * pGlobals->settings->circuit.genomeWidth;
            int offsetFNC = (local_numLayers * 2 + 1) * pGlobals->settings->circuit.genomeWidth;
            unsigned int pos3 = 0;
            unsigned int pos4 = 0;
            int slot = 0;
            while ((pos4 = line.find("]", pos3)) != string::npos) {
                // PARSE ELEMENTS
                string elem = line.substr(pos3, pos4 - pos3 + 1);
                TrimLeadingSpaces(elem);
                TrimTrailingSpaces(elem);

                // CONNECTOR LAYER
                GENOME_ITEM_TYPE conn = (GENOME_ITEM_TYPE) StringToDouble(elem);

                // FUNCTION
                GENOME_ITEM_TYPE fnc = FNC_NOP;
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
//	return PrintCircuitMemory(g, filePath + "memory", usePredictorMask,bPruneCircuit);
	return PrintCircuitMemory(g, filePath, usePredictorMask,bPruneCircuit);
}


int CircuitGenome::PrintCircuitMemory(GAGenome &g, string filePath, unsigned char *usePredictorMask, int bPruneCircuit) {
    int								status = STAT_OK;
    GA1DArrayGenome<GENOME_ITEM_TYPE>&inputGenome = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) g;
    GA1DArrayGenome<GENOME_ITEM_TYPE>  genome(pGlobals->settings->circuit.genomeSize, GACallbacks::evaluator);
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

	// TODO: of pGlobals->settings->main.evaluatorType == EVALUATOR_OUTPUT_CATEGORIES, then different code for C code is needed


	// 1. Prune circuit (if required)
	// 2. Create header (DOT, C)
	// 3. 



	// TODO: move prunning outside
    //
    // PRUNE CIRCUIT IF REQUIRED
    //
    if (pGlobals->settings->outputs.allowPrunning && bPruneCircuit) {
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

    int numMemoryOutputs = (pGlobals->settings->circuit.useMemory) ? pGlobals->settings->circuit.sizeMemory : 0;
    
	//
	// TODO: REFACT: HEADER
	//
    // VISUAL CIRC: INPUTS 
    visualCirc += "digraph EACircuit {\r\n\
rankdir=BT;\r\n\
edge [dir=none];\r\n\
size=\"6,6\";\r\n\
ordering=out;\r\n";

	// CODE CIRCUIT: 
    if (bCodeCircuit) {
		// FUNCTION HEADER
        codeCirc += "int headerCircuit_inputLayerSize = " + toString(pGlobals->settings->circuit.sizeInput) + ";\n";
        codeCirc += "int headerCircuit_outputLayerSize = " + toString(pGlobals->settings->circuit.sizeOutputLayer) + ";\n";
        codeCirc += "\n";
        codeCirc += "static void circuit(unsigned char inputs[";
        codeCirc += toString(pGlobals->settings->testVectors.inputLength);
        codeCirc += "], unsigned char outputs[";
        codeCirc += toString(pGlobals->settings->circuit.sizeOutputLayer);
        codeCirc += "]) {\n";

		// MEMORY INPUTS (if used)
		if (pGlobals->settings->circuit.useMemory) {
            int sectorLength = pGlobals->settings->circuit.sizeInput - pGlobals->settings->circuit.sizeMemory;
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
        int offsetCON = (layer-1) * pGlobals->settings->circuit.genomeWidth;
        int offsetFNC = (layer) * pGlobals->settings->circuit.genomeWidth;

        int numLayerInputs = 0;
    
		//
		// TODO: REFACT: INPUT LAYER
		//
		if (layer == 1) {
			//
			// DRAW NODES IN INPUT LAYER
			//

			// Use magenta color for memory nodes (if used)
			if (pGlobals->settings->circuit.useMemory) visualCirc += "node [color=magenta, style=filled];\r\n";

            for (int i = 0; i < pGlobals->settings->circuit.sizeInput; i++) {
				// set color for data input nodes, when necessary
                if ((pGlobals->settings->circuit.useMemory && (i ==  pGlobals->settings->circuit.sizeMemory)) ||  // all memory inputs already processed
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
                for (int dataInput = pGlobals->settings->circuit.sizeMemory; dataInput < pGlobals->settings->circuit.sizeInput; dataInput++) {
					ostringstream os3;
                    os3 << "        VAR_" <<  "IN_" << dataInput << " = inputs[sector * SECTOR_SIZE + " << dataInput - pGlobals->settings->circuit.sizeMemory << "];\n";
					value2 = os3.str();
					codeCirc += value2;
				}

				codeCirc += "\n";
			}

            numLayerInputs = pGlobals->settings->circuit.sizeInput;
        }
        else numLayerInputs = pGlobals->settings->circuit.sizeLayer;

		visualCirc += "node [color=lightblue2, style=filled];\r\n";

        int numFncs = pGlobals->settings->circuit.sizeLayer;
        // IF DISPLAYING THE LAST LAYER, THEN DISPLAY ONLY 'INTERNAL_LAYER_SIZE' FNC (OTHERS ARE UNUSED)
        if (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) numFncs = pGlobals->settings->circuit.sizeOutputLayer;

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

            GENOME_ITEM_TYPE effectiveCon = genome.gene(offsetCON + slot);
			//FilterEffectiveConnections(genome.gene(offsetFNC + slot), genome.gene(offsetCON + slot), numLayerConnectors, &effectiveCon);
			
			//value2.Format("%.10u[%s]  ", effectiveCon, value);
			// TXT: Transform relative connector mask into absolute mask (fixed inputs from previous layer)
            GENOME_ITEM_TYPE absoluteCon = 0;
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
                    switch (nodeGetFunction(genome.gene(offsetFNC + slot))) {
						case FNC_CONST: {
							//value2.Format("    BYTE VAR_%s = %u", actualSlotID, effectiveCon % UCHAR_MAX);
							ostringstream os7;
                            os7 << "        unsigned char VAR_" << actualSlotID << " = " << (nodeGetArgument1(genome.gene(offsetFNC + slot)) & 0xff);
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
                    //if (HasImplicitConnection(nodeGetFunction(genome.gene(offsetFNC + slot)))) {
					if (false){
						if (layer > 1) {
                            int prevOffsetFNC = (layer - 2) * pGlobals->settings->circuit.genomeWidth;
                            int prevOffsetCON = (layer - 3) * pGlobals->settings->circuit.genomeWidth;
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
	                    					    
                        switch (nodeGetFunction(genome.gene(offsetFNC + slot))) {
							case FNC_NOP: os13 << " VAR_" << previousSlotID; value2 = os13.str(); break; 
							case FNC_SUBS: os13 << " VAR_" << previousSlotID << " - "; value2 = os13.str(); break; 
							case FNC_ADD: os13 << " VAR_" << previousSlotID << " + "; value2 = os13.str(); break; 
							case FNC_MULT: os13 << " VAR_" << previousSlotID << " * "; value2 = os13.str(); break; 
							case FNC_DIV: os13 << " VAR_" << previousSlotID << " / "; value2 = os13.str(); break; 
                            case FNC_ROTL: os13 << " VAR_" << previousSlotID << " << " << (nodeGetArgument1(genome.gene(offsetFNC + slot) & 0x07)); value2 = os13.str(); break;
                            case FNC_ROTR: os13 << " VAR_" << previousSlotID << " >> " << (nodeGetArgument1(genome.gene(offsetFNC + slot) & 0x07)); value2 = os13.str(); break;
                            case FNC_BITSELECTOR: os13 << " VAR_" << previousSlotID + " & " << (nodeGetArgument1(genome.gene(offsetFNC + slot) & 0x07)); value2 = os13.str(); break;
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
			    

				//
				// TODO: REFACT: CONNECTIONS
				//

				for (int bit = 0; bit < stopBit; bit++) {
					// IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE INPUT
					if (HasConnection(genome.gene(offsetFNC + slot), effectiveCon, slot, connectOffset, bit)) {
						//int    bExplicitConnection = TRUE;
						bAtLeastOneConnection = TRUE;
	                    
						if (layer > 1) {
                            int prevOffsetFNC = (layer - 2) * pGlobals->settings->circuit.genomeWidth;
                            int prevOffsetCON = (layer - 3) * pGlobals->settings->circuit.genomeWidth;
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
						//value2.Format("\"%s\" -> \"%s\";\r\n", actualSlotID, previousSlotID);
						ostringstream os23;
						os23 << "\"" << actualSlotID << "\" -> \"" << previousSlotID << "\";\r\n";
						value2 = os23.str();
						visualCirc += value2;
	                    
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
							if (IsOperand(genome.gene(offsetFNC + slot), effectiveCon, slot, connectOffset, bit, &operand)) {
                                if (nodeGetFunction(genome.gene(offsetFNC + slot)) == FNC_DIV) {
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
	// TODO: REFACT: OUTPUT LAYER
	//

	//
	// DRAW OUTPUT LAYER
	//

    // VISUAL CIRC: CONNECT OUTPUT LAYER
    //for (int i = 0; i < pGlobals->settings->circuit.sizeOutputLayer; i++) {

	if (pGlobals->settings->circuit.useMemory) visualCirc += "node [color=magenta];\r\n"; 

	// propagate memory outputs to inputs to next iteration (if required)
    if (pGlobals->settings->circuit.useMemory) {
		// set memory inputs by respective memory outputs 
        for (int memorySlot = 0; memorySlot < pGlobals->settings->circuit.sizeMemory; memorySlot++) {
            int prevOffsetFNC = (2 * pGlobals->settings->circuit.numLayers - 1) * pGlobals->settings->circuit.genomeWidth;
            int prevOffsetCON = (2 * pGlobals->settings->circuit.numLayers - 2) * pGlobals->settings->circuit.genomeWidth;
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
    for (int i = 0; i < pGlobals->settings->circuit.sizeOutputLayer; i++) {
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
            int prevOffsetFNC = (2 * pGlobals->settings->circuit.numLayers - 1) * pGlobals->settings->circuit.genomeWidth;
            int prevOffsetCON = (2 * pGlobals->settings->circuit.numLayers - 2) * pGlobals->settings->circuit.genomeWidth;
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
	// TODO: REFACT: SAVING
	//

	//
    //	ACTUAL WRITING TO DISK
	//
    if (filePath == "") filePath = FILE_CIRCUIT_DEFAULT;

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
    status = CircuitIO::genomeToPopulation(genome,filePath + ".xml");

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

	// Print distribution of categories
    newFilePath = filePath + ".cat";
    file.open(newFilePath.c_str(), fstream::in | fstream::out | fstream::ate | fstream::trunc);
    if (file.is_open()) {
		for (int i = 0; i < NUM_OUTPUT_CATEGORIES; i++) {
			file << setw(3) << i << " ";
			file << setw(15) << pGlobals->testVectors.circuitOutputCategoriesRandom[i] << " ";
			file << setw(15) << pGlobals->testVectors.circuitOutputCategories[i] << " ";
			file << setw(15) << (int) pow(pGlobals->testVectors.circuitOutputCategoriesRandom[i] - pGlobals->testVectors.circuitOutputCategories[i], 2);
			file << endl;
		}
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
    GA1DArrayGenome<GENOME_ITEM_TYPE>&genome = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) g;
//    string							message;
    string							value;
    string							value2;
    string							visualCirc = "";
//    string							codeCirc = "";
	string 							actualSlotID; 
	string 							previousSlotID; 
	int								bCodeCircuit = TRUE; 

    int numMemoryOutputs = (pGlobals->settings->circuit.useMemory) ? pGlobals->settings->circuit.sizeMemory : 0;
    
    // VISUAL CIRC: INPUTS 
    visualCirc += "digraph EACircuit {\r\n\
rankdir=BT;\r\n\
edge [dir=none];\r\n\
size=\"6,6\";\r\n\
ordering=out;\r\n";

    for (int layer = 1; layer < 2 * pGlobals->settings->circuit.numLayers; layer = layer + 2) {
        int offsetCON = (layer-1) * pGlobals->settings->circuit.genomeWidth;
        int offsetFNC = (layer) * pGlobals->settings->circuit.genomeWidth;

        int numLayerInputs = 0;
        if (layer == 1) {
			//
			// DRAW NODES IN INPUT LAYER
			//

			// Use magenta color for memory nodes (if used)
			if (pGlobals->settings->circuit.useMemory) visualCirc += "node [color=magenta, style=filled];\r\n";

            for (int i = 0; i < pGlobals->settings->circuit.sizeInput; i++) {
				// set color for data input nodes, when necessary
                if ((pGlobals->settings->circuit.useMemory && (i ==  pGlobals->settings->circuit.sizeMemory)) ||  // all memory inputs already processed
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
            numLayerInputs = pGlobals->settings->circuit.sizeInput;
        }
        else numLayerInputs = pGlobals->settings->circuit.sizeLayer;

		visualCirc += "node [color=lightblue2, style=filled];\r\n";

        int numFncs = pGlobals->settings->circuit.sizeLayer;
        // IF DISPLAYING THE LAST LAYER, THEN DISPLAY ONLY 'INTERNAL_LAYER_SIZE' FNC (OTHERS ARE UNUSED)
        if (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) numFncs = pGlobals->settings->circuit.sizeOutputLayer;

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

            GENOME_ITEM_TYPE effectiveCon = genome.gene(offsetCON + slot);
			
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
                            int prevOffsetFNC = (layer - 2) * pGlobals->settings->circuit.genomeWidth;
                            int prevOffsetCON = (layer - 3) * pGlobals->settings->circuit.genomeWidth;
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
    //for (int i = 0; i < pGlobals->settings->circuit.sizeOutputLayer; i++) {

	if (pGlobals->settings->circuit.useMemory) visualCirc += "node [color=magenta];\r\n"; 

	// propagate memory outputs to inputs to next iteration (if required)
    if (pGlobals->settings->circuit.useMemory) {
		// set memory inputs by respective memory outputs 
        for (int memorySlot = 0; memorySlot < pGlobals->settings->circuit.sizeMemory; memorySlot++) {
            int prevOffsetFNC = (2 * pGlobals->settings->circuit.numLayers - 1) * pGlobals->settings->circuit.genomeWidth;
            int prevOffsetCON = (2 * pGlobals->settings->circuit.numLayers - 2) * pGlobals->settings->circuit.genomeWidth;
		    string		value;
		    GetFunctionLabel(genome.gene(prevOffsetFNC + memorySlot), genome.gene(prevOffsetCON + memorySlot), &value);
			ostringstream os30;
            os30 << (pGlobals->settings->circuit.numLayers) << "_" << memorySlot << "_" << value;
			previousSlotID = os30.str();
		}
	}

	int outputOffset = 0;
    for (int i = 0; i < pGlobals->settings->circuit.sizeOutputLayer; i++) {
		if (i == numMemoryOutputs) {
			visualCirc += "node [color=red];\r\n"; 
		}
		
		if (true) {
		    //actualSlotID.Format("%d_OUT", i);
			ostringstream os29;
			os29 << i << "_OUT";
			actualSlotID = os29.str();
            int prevOffsetFNC = (2 * pGlobals->settings->circuit.numLayers - 1) * pGlobals->settings->circuit.genomeWidth;
            int prevOffsetCON = (2 * pGlobals->settings->circuit.numLayers - 2) * pGlobals->settings->circuit.genomeWidth;
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
    if (filePath == "") filePath = FILE_CIRCUIT_DEFAULT;

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

void CircuitGenome::executeCircuit(GA1DArrayGenome<GENOME_ITEM_TYPE>* pGenome, unsigned char* inputs, unsigned char* outputs) {
//    unsigned char*   inputsBegin = inputs;
    int     numSectors = 1;
    int     sectorLength = pGlobals->settings->circuit.sizeInput;
    int     memoryLength = 0;
    unsigned char*    localInputs = NULL;
    unsigned char*    localOutputs = NULL;
    unsigned char*    fullLocalInputs = NULL;

	// Compute maximum number of inputs into any layer
    //int maxLayerSize = (pGlobals->settings->circuit.sizeInputLayer > pGlobals->settings->circuit.sizeLayer) ? pGlobals->settings->circuit.sizeInputLayer : pGlobals->settings->circuit.sizeLayer;
    int maxLayerSize = max(pGlobals->settings->circuit.sizeInputLayer, pGlobals->settings->circuit.genomeWidth);

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
        // sizeof(MEMORY_INPUTS_i) == pGlobals->settings->circuit.memorySize (0 if useMemory = false)
		// sizeof(DATA_INPUTS_i) == pGlobals->settings->circuit.sizeInputLayer - pGlobals->settings->circuit.memorySize
        // wanted: sizeof(DATA_INPUTS_i) == pGlobals->settings->circuit.sizeInput
		// sizeof(MEMORY_INTPUTS_i+1) == pGlobals->settings->circuit.memorySize
		// sizeof(DATA_OUTPUTS_i) == pGlobals->settings->circuit.sizeOutputLayer
        // wanted: sizeof(DATA_OUTPUTS_i) == pGlobals->settings->circuit.sizeOutput

        sectorLength = pGlobals->settings->circuit.sizeInput - pGlobals->settings->circuit.sizeMemory;
        memoryLength = pGlobals->settings->circuit.sizeMemory;
		numSectors = pGlobals->settings->testVectors.inputLength / sectorLength;

		assert(pGlobals->settings->testVectors.inputLength % sectorLength == 0);
	}
	else {
		// ALL IN ONE RUN
		numSectors = 1;
		memoryLength = 0;
        sectorLength = pGlobals->settings->circuit.sizeInput;
	}

#ifdef ENABLE_SLIDING_WINDOW
	// Inputs is not partitioned into separate sectors with pGlobals->settings->circuit.sizeInputLayer length, but with overlapping parts with pGlobals->settings->circuit.sizeInputLayer length
    for (float sector = 0; sector < numSectors; sector += 1 / (float) sectorLength) { 
#else
    for (int sector = 0; sector < numSectors; sector++) { 
#endif
        // PREPARE INPUTS FOR ACTUAL RUN OF CIRCUIT
        if (numSectors == 1) {
            // ALL INPUT DATA AT ONCE
            memcpy(localInputs, inputs, pGlobals->settings->circuit.sizeInput);
			// duplicate before and after (see TRICK above)
            memcpy(localInputs - pGlobals->settings->circuit.sizeInput, localInputs, pGlobals->settings->circuit.sizeInput);
            memcpy(localInputs + pGlobals->settings->circuit.sizeInput, localInputs, pGlobals->settings->circuit.sizeInput);
        }
        else {
            // USE MEMORY STATE (OUTPUT) AS FIRST PART OF INPUT
			// NOTE: for first iteration, memory is zero (taken from localOutputs)
            memcpy(localInputs, localOutputs, memoryLength);
            // ADD FRESH INPUT DATA
			int inputOffset = sector * sectorLength;
            memcpy(localInputs + memoryLength, inputs + inputOffset, sectorLength);
			int realInputsLength = memoryLength + sectorLength;
			assert(realInputsLength <= maxLayerSize);
			// duplicate before and after
			memcpy(localInputs - realInputsLength, localInputs, realInputsLength);
			memcpy(localInputs + realInputsLength, localInputs, realInputsLength);
        }
        
        // EVALUATE CIRCUIT
        for (int layer = 1; layer < 2 * pGlobals->settings->circuit.numLayers; layer = layer + 2) {
            int offsetCON = (layer-1) * pGlobals->settings->circuit.genomeWidth;
            int offsetFNC = (layer) * pGlobals->settings->circuit.genomeWidth;
            memset(localOutputs, 0, maxLayerSize); // BUGBUG: can be sizeInputLayer lower than number of used items in localOutputs?

            // actual number of inputs for this layer. For first layer equal to pCircuit->numInputs, for next layers equal to number of function in intermediate layer pCircuit->internalLayerSize
            int numLayerInputs = 0;
            if (layer == 1) {
                // WHEN SECTORING IS USED THEN FIRST LAYER OBTAIN sizeInputLayer INPUTS
                numLayerInputs = pGlobals->settings->circuit.sizeInput;
            }
            else numLayerInputs = pGlobals->settings->circuit.sizeLayer;

            // actual number of functions in layer - different for the last "output" layer
            int numFncInLayer = (layer == (2 * pGlobals->settings->circuit.numLayers - 1)) ? (pGlobals->settings->circuit.sizeOutputLayer) : pGlobals->settings->circuit.sizeLayer;

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
                GENOME_ITEM_TYPE   connect = 0;
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
                switch (nodeGetFunction(pGenome->gene(offsetFNC + slot))) {
                    case FNC_NOP: {
                        // DO NOTHING, JUST PASS VALUE FROM FIRST CONNECTOR FROM PREVIOUS LAYER
                        for (int bit = 0; bit < stopBit; bit++) {
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
								result = localInputs[connectOffset + bit];
								break; // pass only one value
                            }
                        }
                        break;
                    }
                    case FNC_OR: {
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE INPUT
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result |= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_AND: {
                        result = ~0; // ASSIGN ALL ONES AS STARTING VALUE TO PERFORM result &= LATER
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION, THEN TAKE INPUT
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result &= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_CONST: {
                        // SEND VALUE FROM CONNECTION LAYER DIRECTLY TO OUTPUT
                        result = nodeGetArgument1(pGenome->gene(offsetCON + slot));
                        break;
                    }
                    case FNC_XOR: {
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE INPUT
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result ^= localInputs[connectOffset + bit];
                            }
                        }
                        break;
                    }
                    case FNC_NOR: {
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE NEGATION OF INPUT
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result |= ~(localInputs[connectOffset + bit]);
                            }
                        }
                        break;
                    }
                    case FNC_NAND: {
                        result = ~0; // ASSIGN ALL ONES AS STARTING VALUE TO PERFORM result &= LATER
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TAKE NEGATION OF INPUT
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result &= ~(localInputs[connectOffset + bit]);
                            }
                        }
                        break;
					}
                    case FNC_ROTL: {
                        // SHIFT IS ENCODED IN FUNCTION IDENTFICATION 
						// MAXIMUM SHIFT IS 7
                        for (int bit = 0; bit < stopBit; bit++) {
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result = localInputs[connectOffset + bit] << (nodeGetArgument1(pGenome->gene(offsetFNC + slot) & 0x07));
								break; // pass only one value
                            }
                        }
                        
                        break;
                    }
                    case FNC_ROTR: {
                        // SHIFT IS ENCODED IN FUNCTION IDENTFICATION 
						// MAXIMUM SHIFT IS 7
                        for (int bit = 0; bit < stopBit; bit++) {
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result = localInputs[connectOffset + bit] >> (nodeGetArgument1(pGenome->gene(offsetFNC + slot) & 0x07));
								break; // pass only one value
                            }
                        }
                        
                    }
                    case FNC_BITSELECTOR: {
                        // BIT SELECTOR
                        // MASK IS ENCODED IN FUNCTION IDENTFICATION 
                        for (int bit = 0; bit < stopBit; bit++) {
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
                                result = localInputs[connectOffset + bit] & (nodeGetArgument1(pGenome->gene(offsetFNC + slot) & 0x07));
								break; // pass only fisrt value
                            }
                        }
                        
                        break;
                    }
                    case FNC_SUM: {
                        // SUM ALL INPUTS
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN SUM IT INTO OF INPUT
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
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
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
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
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
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
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
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
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
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
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
								result += localInputs[connectOffset + bit];
							}
						}
                        result = inputs[result % pGlobals->settings->circuit.sizeInput];
                        break;
					}
					case FNC_EQUAL: {
                        // COMPARE ALL INPUTS
						bool bFirstInput = true;
						unsigned char firstValue = 0;
						result = 1; // assume equalilty of inputs
                        for (int bit = 0; bit < stopBit; bit++) {
                            // IF 1 IS ON bit-th POSITION (CONNECTION LAYER), THEN TEST EQUALITY OF INPUTS
                            if (connect & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
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

		// Compute categories as classified by the circuit - take only first output, add to histogram of outputs
		// Shrink number of categories down to NUM_OUTPUT_CATEGORIES
		pGlobals->testVectors.circuitOutputCategories[localOutputs[memoryLength] % NUM_OUTPUT_CATEGORIES] += 1;
    }
    
	// circuit output is taken from output parts AFTER memory outputs
	// BUT length of memory outputs can be zero (in case of circuit without memory)
    memcpy(outputs,localOutputs + memoryLength,pGlobals->settings->circuit.sizeOutput);

	delete[] fullLocalInputs;
    delete[] localOutputs;
}

/*
int CircuitGenome::writeGenome(const GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string& textCircuit) {
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

int CircuitGenome::saveCircuitAsPopulation(const GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, const string filename) {
    int status = STAT_OK;
    TiXmlElement* pRoot = CircuitGenome::populationHeader(1);
    TiXmlElement* pElem = NULL;
    TiXmlElement* pElem2 = NULL;

    pElem = new TiXmlElement("population");
    string textCircuit;
    GA1DArrayGenome<GENOME_ITEM_TYPE>* pGenome = (GA1DArrayGenome<GENOME_ITEM_TYPE>*) &genome;
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
*/
