#ifndef _POLYNOMIAL_INTERPRETER_H
#define _POLYNOMIAL_INTERPRETER_H
#include "PolyCommonFunctions.h"

class PolynomialInterpreter {
public:
    /** Evaluate the distinguisher polynomial encoded in the bare form on given inputs, return outputs.
     * @param pGenome       polynomial for computation
     * @param inputs        input data
     * @param outputs       produced output
     * @return status
     */
    static int executePolynomial(GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>* pGenome, unsigned char* inputs, unsigned char* outputs);

    /**
     * Converts to the ANF (using number sort on the terms).
     * Reduces duplicities, preserves polynomial function.
     *
     * @param pGenome
     * @return
     */
    static int normalize(GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>* pGenome);
};

#endif // _POLYNOMIAL_INTERPRETER_H
