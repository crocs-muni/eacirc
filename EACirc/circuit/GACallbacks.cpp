#include "GACallbacks.h"
#include "EACglobals.h"
#include "evaluators/IEvaluator.h"
#include "generators/IRndGen.h"
#include "CircuitGenome.h"

void GACallbacks::initializer(GAGenome& genome) {
    GA1DArrayGenome<GENOM_ITEM_TYPE>& g = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) genome;
    GACallbacks::initializer_basic(g);
}

float GACallbacks::evaluator(GAGenome &g) {
    GA1DArrayGenome<GENOM_ITEM_TYPE>  &genome = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) g;
    // reset evaluator state for this individual
    pGlobals->evaluator->resetEvaluator();
    // execute circuit & evaluate success for each test vector
    for (int testVector = 0; testVector < pGlobals->settings->testVectors.setSize; testVector++) {
        CircuitGenome::executeCircuit(&genome, pGlobals->testVectors.inputs[testVector], pGlobals->testVectors.circuitOutputs[testVector]);
        pGlobals->evaluator->evaluateCircuit(pGlobals->testVectors.circuitOutputs[testVector], pGlobals->testVectors.outputs[testVector]);
    }
    // retrieve fitness from evaluator
    return pGlobals->evaluator->getFitness();
}

int GACallbacks::mutator(GAGenome &genome, float probMutation) {
    GA1DArrayGenome<GENOM_ITEM_TYPE> &g = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) genome;
    return mutator_basic(g, probMutation);
}

int GACallbacks::crossover(const GAGenome &parent1, const GAGenome &parent2, GAGenome *offspring1, GAGenome *offspring2) {
    GA1DArrayGenome<GENOM_ITEM_TYPE> &p1 = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) parent1;
    GA1DArrayGenome<GENOM_ITEM_TYPE> &p2 = (GA1DArrayGenome<GENOM_ITEM_TYPE>&) parent2;
    GA1DArrayGenome<GENOM_ITEM_TYPE>* o1 = (GA1DArrayGenome<GENOM_ITEM_TYPE>*) offspring1;
    GA1DArrayGenome<GENOM_ITEM_TYPE>* o2 = (GA1DArrayGenome<GENOM_ITEM_TYPE>*) offspring2;
    return crossover_perColumn(p1, p2, o1, o2);
    //return crossover_perLayer(p1, p2, o1, o2);
}

void GACallbacks::initializer_basic(GA1DArrayGenome<GENOM_ITEM_TYPE>& genome) {

    // clear genome
    for (int i = 0; i < genome.size(); i++) genome.gene(i, 0);

    // 1. IN_SELECTOR_LAYER connects inputs to corresponding FNC in the same column (FNC_1_3->IN_3)
    int offset = 0;
    for (int slot = 0; slot < pGlobals->settings->circuit.sizeLayer; slot++){
        // NOTE: relative mask must be computed (in relative masks, (numLayerInputs / 2) % numLayerInputs is node at same column)
        genome.gene(offset + slot, pGlobals->precompPow[(pGlobals->settings->circuit.sizeInputLayer / 2) % pGlobals->settings->circuit.sizeInputLayer]);
    }

    // 2. FUNCTION_LAYER_1 is set to XOR instruction only
    offset = 1 * pGlobals->settings->circuit.sizeLayer;
    for (int slot = 0; slot < pGlobals->settings->circuit.sizeLayer; slot++) {
        genome.gene(offset + slot, FNC_XOR);
    }

    // 3. CONNECTOR_LAYER_i is set to random mask (possibly multiple connectors)
    for (int layer = 1; layer < pGlobals->settings->circuit.numLayers; layer++) {
        offset = (2 * layer) * pGlobals->settings->circuit.sizeLayer;
        int layerSize = layer == pGlobals->settings->circuit.numLayers ? pGlobals->settings->circuit.sizeLayer : pGlobals->settings->circuit.sizeOutputLayer;
        for (int slot = 0; slot < layerSize; slot++) {
            // TBD: what does numConnectors mean? number of connectors, or their maximum distance or both?
            // TBD: make connector number generation correct
            genome.gene(offset + slot, GARandomInt(0,pGlobals->precompPow[pGlobals->settings->circuit.numConnectors]));
        }
    }

    // 4. FUNCTION_LAYER_i is set to random instruction from range 0..FNC_MAX, respecting allowed instructions in settings
    for (int layer = 1; layer < pGlobals->settings->circuit.numLayers; layer++) {
        offset = (2 * layer + 1) * pGlobals->settings->circuit.sizeLayer;
        int layerSize = layer == pGlobals->settings->circuit.numLayers ? pGlobals->settings->circuit.sizeLayer : pGlobals->settings->circuit.sizeOutputLayer;
        for (int slot = 0; slot < layerSize; slot++) {
            int function;
            do {
                function = GARandomInt(0,FNC_MAX);
                genome.gene(offset + slot, function);
            } while (pGlobals->settings->circuit.allowedFunctions[function] != 0);
        }
    }

    /*
    // ***
    int	offset = 0;

    // CLEAR GENOME
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
                    value = GARandomInt(0,pGlobals->precompPow[pGlobals->settings->circuit.numConnectors]);
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
                        value = GARandomInt(0,FNC_MAX);
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
    */
}

int GACallbacks::mutator_basic(GA1DArrayGenome<GENOM_ITEM_TYPE>& genome, float probMutation) {
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
                if (GAFlipCoin(probMutation)) {
                    GENOM_ITEM_TYPE temp;
                    value = GARandomInt(0,numLayerInputs);
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
                if (GAFlipCoin(probMutation)) {
                    unsigned char newFncValue = 0;
                    int bFNCNotSet = TRUE;
                    while (bFNCNotSet) {
                        newFncValue = GARandomInt(0, FNC_MAX);
                        if (pGlobals->settings->circuit.allowedFunctions[newFncValue] != 0) {
                            CircuitGenome::SET_FNC_TYPE(&origValue, newFncValue);
                            bFNCNotSet = FALSE;
                        }
                    }
                    genome.gene(offset + slot, origValue);
                }
                // MUTATE FNC ARGUMENT1
                if (GAFlipCoin(probMutation)) {
                    origValue = genome.gene(offset + slot);

                    unsigned char newArg1Value = 0;
                    newArg1Value = GARandomInt(0, ULONG_MAX);
                    // SWITCH RANDOMLY GENERATED BITS
                    newArg1Value = newArg1Value ^ CircuitGenome::GET_FNC_ARGUMENT1(genome.gene(offset + slot));
                    CircuitGenome::SET_FNC_ARGUMENT1(&origValue, newArg1Value);
                    genome.gene(offset + slot, origValue);
                }
            }
        }
    }

    return result;
}

int GACallbacks::crossover_perLayer(const GA1DArrayGenome<GENOM_ITEM_TYPE> &parent1, const GA1DArrayGenome<GENOM_ITEM_TYPE> &parent2,
                                    GA1DArrayGenome<GENOM_ITEM_TYPE> *offspring1, GA1DArrayGenome<GENOM_ITEM_TYPE> *offspring2) {
    // take random layer and cut individuals in 2 parts horisontally
    int crossPoint = GARandomInt(1,pGlobals->settings->circuit.numLayers) * 2;
    for (int layer = 0; layer < 2 * pGlobals->settings->circuit.numLayers; layer++) {
        int offset = layer * pGlobals->settings->circuit.sizeLayer;
        if (layer <= crossPoint) {
            for (int i = 0; i < pGlobals->settings->circuit.sizeLayer; i++) {
                if (offspring1 != NULL) offspring1->gene(offset + i, parent1.gene(offset + i));
                if (offspring2 != NULL) offspring2->gene(offset + i, parent2.gene(offset + i));
            }
        } else {
            for (int i = 0; i < pGlobals->settings->circuit.sizeLayer; i++) {
                if (offspring1 != NULL) offspring1->gene(offset + i, parent2.gene(offset + i));
                if (offspring2 != NULL) offspring2->gene(offset + i, parent1.gene(offset + i));
            }
        }
    }
    return 1;
}
int GACallbacks::crossover_perColumn(const GA1DArrayGenome<GENOM_ITEM_TYPE> &parent1, const GA1DArrayGenome<GENOM_ITEM_TYPE> &parent2,
                                     GA1DArrayGenome<GENOM_ITEM_TYPE> *offspring1, GA1DArrayGenome<GENOM_ITEM_TYPE> *offspring2) {
    // take random point in layer size and cut individuals in 2 parts verically
    int crossPoint = GARandomInt(1,pGlobals->settings->circuit.sizeLayer);
    for (int layer = 0; layer < 2 * pGlobals->settings->circuit.numLayers; layer++) {
        int offset = layer * pGlobals->settings->circuit.sizeLayer;
        for (int i = 0; i < pGlobals->settings->circuit.sizeLayer; i++) {
            if ( i < crossPoint) {
                if (offspring1 != NULL) offspring1->gene(offset + i, parent1.gene(offset + i));
                if (offspring2 != NULL) offspring2->gene(offset + i, parent2.gene(offset + i));
            } else {
                if (offspring1 != NULL) offspring1->gene(offset + i, parent2.gene(offset + i));
                if (offspring2 != NULL) offspring2->gene(offset + i, parent1.gene(offset + i));
            }
        }
    }
    return 1;
}
