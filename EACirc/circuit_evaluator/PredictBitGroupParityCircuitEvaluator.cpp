#include "PredictBitGroupParityCircuitEvaluator.h"

PredictBitGroupParityCircuitEvaluator::PredictBitGroupParityCircuitEvaluator() : ICircuitEvaluator(){
}

void PredictBitGroupParityCircuitEvaluator::evaluateCircuit(unsigned char* outputs, unsigned char* correctOutputs, unsigned char* usePredictorsMask, int* pMatch, int* pTotalPredictCount, int* predictorMatch = NULL){
	// OUTPUT LAYER ENCODES THE GROUP OF BITS AND THEIR PARITY
    // SYNTAX: EACH BYTE B OF OUTPUT ENCODE POSITION OF ONE SINGLE BIT OF OUTPUT, IF B IS EVEN, THEN IS NOT USED 
    // PARITY OF FIRST BYTE OF OUTPUT IS PREDICTED PARITY OF GROUP OF BYTES 
    // COUNT NUMBER OF CORRECT BITS 
    int predictParity = 0;
    int correctParity = 0;
    int offsetBlock = 0;
    int offsetBit = 0;
    int numBitsPredicted = 0;
            
    unsigned char usedBits[MAX_OUTPUTS * BITS_IN_UCHAR];
    memset(usedBits, 0, sizeof(usedBits));
            
    // GET PREDICTION OF PARITY (parity of first output bit) 
    for (int bit = 0; bit < BITS_IN_UCHAR; bit++) {
        if (outputs[0] & (unsigned char) pGlobals->precompPow[bit]) predictParity++;
    }
    predictParity = (predictParity & 0x01) ? 1 : 0; 
            
    // COMPUTE REAL PAIRITY OF SIGNALIZED BITS
    for (int out = 1; out < pGlobals->settings->circuit.sizeOutputLayer; out++) {
        if (outputs[out] & 0x80) { // mask out highest bit (used/not used flag)
            // THIS OUTPUT WILL BE USED 
            // OBTAIN VALUE OF ENCODED BIT
            offsetBit = outputs[out] & 0x7f;  // take all bits except used/not used bit
                    
            // CHECK IF THIS BIT WAS NOT ALREADY USED AND DO NOT EXCEED THE NUMBER OF BITS IN OUTPUT
            if (usedBits[offsetBit] == 0 && (offsetBit < pGlobals->settings->circuit.sizeOutputLayer * BITS_IN_UCHAR)) {
                // MARK THIS BIT AS USED TO PREVENT MULTIPLE SELECTION OF THE SAME BITS
                usedBits[offsetBit] = 1;
                        
                offsetBlock = (int) offsetBit / BITS_IN_UCHAR;
                offsetBit = offsetBit - (offsetBlock * BITS_IN_UCHAR);
                assert(offsetBit < BITS_IN_UCHAR);
                assert(offsetBlock < pGlobals->settings->circuit.sizeOutputLayer);

                if (correctOutputs[offsetBlock] & (unsigned char) pGlobals->precompPow[offsetBit]) {
                    // SIGNALIZED BIT HAS VALUE 1
                    correctParity++;
                }
                        
                // AT LEAST ONE BIT WAS PREDICTED
                numBitsPredicted++;
            }
        }
        else {
            // NOT USED
        }
    }
    correctParity = (correctParity & 0x01) ? 1 : 0; 

    // DO NOT ALLOW SOLUTIONS THAT DO PREDICT NO BITS
    if (numBitsPredicted >= 1) {
                
		if (predictParity == correctParity) {
			(*pMatch)++;      
		}
        (*pTotalPredictCount)++;                    
    }
}
