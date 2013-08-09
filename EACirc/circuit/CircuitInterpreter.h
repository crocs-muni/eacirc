#ifndef CIRCUITINTERPRETER_H
#define CIRCUITINTERPRETER_H

#include "CircuitCommonFunctions.h"

class CircuitInterpreter {
public:
    /** execute circuit over given inputs, return outputs
     * @param pGenome       circuit to executeCircuit
     * @param inputs
     * @param outputs
     * @return status
     */
    static int executeCircuit(GA1DArrayGenome<GENOME_ITEM_TYPE>* pGenome, unsigned char* inputs, unsigned char* outputs);

    /** prunde circuit (delete nodes and connectors not used in actual fitness computation)
     * @param originalGenome
     * @param prunnedGenome
     * @return status
     */
    static int pruneCircuit(GAGenome &originalGenome, GAGenome &prunnedGenome);
};

#endif // CIRCUITINTERPRETER_H
