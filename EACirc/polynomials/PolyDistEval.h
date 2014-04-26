#ifndef _EACIRC_POLYEVAL_H
#define _EACIRC_POLYEVAL_H
#include "poly.h"

class PolyEval {
public:
    /** 
     * Evaluate the distinguisher encoded in the bare form on given inputs, return outputs.
     * 
     * @param pGenome       circuit to executeCircuit
     * @param inputs
     * @param outputs
     * @return status
     */
    static int polyEval(GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>* pGenome, unsigned char* inputs, unsigned char* outputs);
    
    /**
     * Converts to the ANF (using number sort on the terms).
     * Reduces duplicities, preserves polynomial function.
     * 
     * @param pGenome
     * @return 
     */
    static int normalize(GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>* pGenome);
};

#endif // _EACIRC_POLYEVAL_H
