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

private:
    /** compute single circuit function
     * @param node                  function constant (with optional arguments)
     * @param absoluteConnectors    connector mask (absolute!)
     * @param layerInputValues      array of values produced by previous layer
     * @param result                computed value
     * @return status
     */
    static int executeFunction(GENOME_ITEM_TYPE node, GENOME_ITEM_TYPE absoluteConnectors, unsigned char* layerInputValues, unsigned char &result);

    /** compute single external circuit function
     * @param node                  function constant (with optional arguments)
     * @param absoluteConnectors    connector mask (absolute!)
     * @param layerInputValues      array of values produced by previous layer
     * @param result                computed value
     * @return status
     */
    static int executeExternalFunction(GENOME_ITEM_TYPE node, GENOME_ITEM_TYPE absoluteConnectors, unsigned char* layerInputValues, unsigned char &result);
};

#endif // CIRCUITINTERPRETER_H
