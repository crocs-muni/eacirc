#ifndef GATE_INTERPRETER_H
#define GATE_INTERPRETER_H

#include "GateCommonFunctions.h"

// temporary arrays for executeCircuit (to prevent multiple allocations)
extern unsigned char* executionInputLayer;     //! input layer (memory + inputs)
extern unsigned char* executionMiddleLayerIn;  //! common layer used as input
extern unsigned char* executionMiddleLayerOut; //! common layer used as output
extern unsigned char* executionOutputLayer;    //! output layer (memoty + outputs)

class GateInterpreter {
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

#endif // GATE_INTERPRETER_H
