#include "GAGateCallbacks.h"
#include "EACglobals.h"
#include "evaluators/IEvaluator.h"
#include "generators/IRndGen.h"
#include "GateInterpreter.h"
#include "GateCommonFunctions.h"

void GAGateCallbacks::initializer(GAGenome& genome) {
    GA1DArrayGenome<GENOME_ITEM_TYPE>& g = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) genome;
    GAGateCallbacks::initializer_basic(g);
}

float GAGateCallbacks::evaluator(GAGenome &g) {
    GA1DArrayGenome<GENOME_ITEM_TYPE>  &genome = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) g;
    // reset evaluator state for this individual
    pGlobals->evaluator->resetEvaluator();
    // execute circuit & evaluate success for each test vector
    for (int testVector = 0; testVector < pGlobals->settings->testVectors.setSize; testVector++) {
        GateInterpreter::executeCircuit(&genome, pGlobals->testVectors.inputs[testVector], pGlobals->testVectors.circuitOutputs[testVector]);
        pGlobals->evaluator->evaluateCircuit(pGlobals->testVectors.circuitOutputs[testVector], pGlobals->testVectors.outputs[testVector]);
    }
    // retrieve fitness from evaluator
    return pGlobals->evaluator->getFitness();
}

int GAGateCallbacks::mutator(GAGenome &genome, float probMutation) {
    GA1DArrayGenome<GENOME_ITEM_TYPE> &g = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) genome;
    return mutator_basic(g, probMutation);
}

int GAGateCallbacks::crossover(const GAGenome &parent1, const GAGenome &parent2, GAGenome *offspring1, GAGenome *offspring2) {
    GA1DArrayGenome<GENOME_ITEM_TYPE> &p1 = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) parent1;
    GA1DArrayGenome<GENOME_ITEM_TYPE> &p2 = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) parent2;
    GA1DArrayGenome<GENOME_ITEM_TYPE>* o1 = (GA1DArrayGenome<GENOME_ITEM_TYPE>*) offspring1;
    GA1DArrayGenome<GENOME_ITEM_TYPE>* o2 = (GA1DArrayGenome<GENOME_ITEM_TYPE>*) offspring2;
    return crossover_perColumn(p1, p2, o1, o2);
    //return crossover_perLayer(p1, p2, o1, o2);
}

void GAGateCallbacks::initializer_basic(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome) {

    // clear genome
    for (int i = 0; i < genome.size(); i++) genome.gene(i, 0);

    // 1. CONNECTOR_LAYER_1 connects inputs to corresponding FNC in the same column (FNC_1_3->IN_3)
    int offset = 0;
    for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeLayer; slot++){
        // for details see connectors documentation
        genome.gene(offset + slot, pGlobals->precompPow[pGlobals->settings->gateCircuit.sizeInputLayer / 2]);
    }

    // 2. FUNCTION_LAYER_1 is set to XOR instruction only (argument random 0-255)
    offset = 1 * pGlobals->settings->gateCircuit.genomeWidth;
    for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeLayer; slot++) {
        GENOME_ITEM_TYPE genomeItem = 0;
        nodeSetFunction(genomeItem, FNC_XOR);
        for (int argIndex = 1; argIndex <= NUM_FNC_ARGUMENTS; argIndex++) {
            nodeSetArgument(genomeItem, argIndex, GARandomInt(0, UCHAR_MAX));
        }
        genome.gene(offset + slot, genomeItem);
    }

    // 3. CONNECTOR_LAYER_i is set to random mask (possibly multiple connectors)
    for (int layer = 1; layer < pGlobals->settings->gateCircuit.numLayers - 1; layer++) {
        offset = (2 * layer) * pGlobals->settings->gateCircuit.genomeWidth;
        for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeLayer; slot++) {
            // for details see connector documentation
            genome.gene(offset + slot, GARandomInt(0,pGlobals->precompPow[pGlobals->settings->gateCircuit.numConnectors]-1));
        }
    }

    // 4. FUNCTION_LAYER_i is set to random instruction from range 0..FNC_MAX, respecting allowed instructions in settings
    for (int layer = 1; layer < pGlobals->settings->gateCircuit.numLayers - 1; layer++) {
        offset = (2 * layer + 1) * pGlobals->settings->gateCircuit.genomeWidth;
        for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeLayer; slot++) {
            GENOME_ITEM_TYPE genomeItem = 0;
            unsigned char function;
            do {
                function = GARandomInt(0,FNC_MAX);
            } while (pGlobals->settings->gateCircuit.allowedFunctions[function] == 0);
            nodeSetFunction(genomeItem, function);

            if (function == FNC_JVM) {
                //set JVM arguments
                nodeSetJVMArguments(genomeItem);
            } else {
                // set argument1 to random value (0-255)
                for (int argIndex = 1; argIndex <= NUM_FNC_ARGUMENTS; argIndex++) {
                    nodeSetArgument(genomeItem, argIndex, GARandomInt(0, UCHAR_MAX));
                }
            }

            genome.gene(offset + slot, genomeItem);
        }
    }

    int layer = pGlobals->settings->gateCircuit.numLayers - 1;
    // 5. last CONNECTOR_LAYER connects to random nodes (not respecting numConnectors!)
    offset = (2 * layer) * pGlobals->settings->gateCircuit.genomeWidth;
    for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeOutputLayer; slot++){
        // for details see connectors documentation
        // BEWARE: relative location of nodes further than sizeLayer are wrapped around (see documentation)
        genome.gene(offset + slot, GARandomInt(0, pGlobals->precompPow[pGlobals->settings->gateCircuit.sizeLayer]-1));
    }

    // 6. last FUNCTION_LAYER is set to random instruction from range 0..FNC_MAX, respecting allowed instructions in settings
    offset = (2 * layer + 1) * pGlobals->settings->gateCircuit.genomeWidth;
    for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeOutputLayer; slot++) {
        GENOME_ITEM_TYPE genomeItem = 0;
        unsigned char function;
        do {
            function = GARandomInt(0,FNC_MAX);
        } while (pGlobals->settings->gateCircuit.allowedFunctions[function] == 0);
        nodeSetFunction(genomeItem, function);
        
        if (function == FNC_JVM) {
            //set JVM arguments
            nodeSetJVMArguments(genomeItem);
        } else {
            // set argument1 to random value (0-255)
            for (int argIndex = 1; argIndex <= NUM_FNC_ARGUMENTS; argIndex++) {
                nodeSetArgument(genomeItem, argIndex, GARandomInt(0, UCHAR_MAX));
            }
        }

        genome.gene(offset + slot, genomeItem);
    }

}

int GAGateCallbacks::mutator_basic(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, float probMutation) {
    int numOfMutations = 0;
    int offset;
    //
    // MUTATE CONNECTORS
    //
    if (pGlobals->settings->ga.mutateConnectors) {
        // mutate connectors in input layer
        offset = 0;
        for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeLayer; slot++){
            if (GAFlipCoin(probMutation)) {
                numOfMutations++;
                // allow any bit 0-sizeInputLayer to change
                genome.gene(offset + slot, changeBit(genome.gene(offset+slot), pGlobals->settings->main.circuitSizeInput));
            }
        }
        // mutate connectors in internal connector layers
        for (int layer = 1; layer < pGlobals->settings->gateCircuit.numLayers - 1; layer++) {
            offset = (2 * layer) * pGlobals->settings->gateCircuit.genomeWidth;
            for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeLayer; slot++){
                if (GAFlipCoin(probMutation)) {
                    numOfMutations++;
                    // allow only bits 0-numConnectors to change
                    genome.gene(offset + slot, changeBit(genome.gene(offset+slot), pGlobals->settings->gateCircuit.numConnectors));
                }
            }
        }
        // mutate connectors in last layer
        offset = (2 * (pGlobals->settings->gateCircuit.numLayers - 1)) * pGlobals->settings->gateCircuit.genomeWidth;
        for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeOutputLayer; slot++){
            if (GAFlipCoin(probMutation)) {
                numOfMutations++;
                // allow any bit 0-sizeLayer to change
                // BEWARE: relative location of nodes further than sizeLayer are wrapped around (see documentation)
                genome.gene(offset + slot, changeBit(genome.gene(offset+slot), pGlobals->settings->gateCircuit.sizeLayer));
            }
        }
    }
    //
    // MUTATE FUNCTIONS AND ARGUMENTS
    //
    if (pGlobals->settings->ga.mutateFunctions) {
        // mutate functions in input layer or internal layers
        for (int layer = 0; layer < pGlobals->settings->gateCircuit.numLayers - 1; layer++) {
            offset = (2 * layer + 1) * pGlobals->settings->gateCircuit.genomeWidth;
            for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeLayer; slot++){
                if (GAFlipCoin(probMutation)) { // mutate function
                    numOfMutations++;
                    unsigned char function;
                    do {
                        function = GARandomInt(0, FNC_MAX);
                    } while (pGlobals->settings->gateCircuit.allowedFunctions[function] == 0);
                    GENOME_ITEM_TYPE genomeItem = genome.gene(offset + slot);
                    nodeSetFunction(genomeItem, function);
                    //JVM arguments has to be valid
                    if (function == FNC_JVM){
                        nodeSetJVMArguments(genomeItem);
                    }
                    genome.gene(offset + slot, genomeItem);
                }
                for (int argIndex = 1; argIndex <= NUM_FNC_ARGUMENTS; argIndex++) {
                    if (GAFlipCoin(probMutation)) { // muatate argument
                        numOfMutations++;
                        GENOME_ITEM_TYPE genomeItem = genome.gene(offset + slot);
                        
                        if (nodeGetFunction(genomeItem) == FNC_JVM){
                            //set JVM arguments
                            //TODO: change only one argument
                            nodeSetJVMArguments(genomeItem);
                        } else{
                            // set argument1 to random value (0-255)
                            nodeSetArgument(genomeItem, argIndex, GARandomInt(0, UCHAR_MAX));
                        }

                        genome.gene(offset + slot, genomeItem);
                    }
                }
            }
        }
        // mutate function in last layer
        offset = (2 * (pGlobals->settings->gateCircuit.numLayers-1) + 1) * pGlobals->settings->gateCircuit.genomeWidth;
        for (int slot = 0; slot < pGlobals->settings->gateCircuit.sizeOutputLayer; slot++){
            if (GAFlipCoin(probMutation)) { // mutate function
                numOfMutations++;
                unsigned char function;
                do {
                    function = GARandomInt(0, FNC_MAX);
                } while (pGlobals->settings->gateCircuit.allowedFunctions[function] == 0);
                GENOME_ITEM_TYPE genomeItem = genome.gene(offset + slot);
                nodeSetFunction(genomeItem, function);

                //JVM arguments has to be valid
                if (function == FNC_JVM){
                    nodeSetJVMArguments(genomeItem);
                }

                genome.gene(offset + slot, genomeItem);
            }
            for (int argIndex = 1; argIndex <= NUM_FNC_ARGUMENTS; argIndex++) {
                if (GAFlipCoin(probMutation)) { // muatate argument
                    numOfMutations++;
                    // set argument1 to random value (0-255)
                    GENOME_ITEM_TYPE genomeItem = genome.gene(offset + slot);
                    
                    if (nodeGetFunction(genomeItem) == FNC_JVM){
                        //set JVM arguments
                        //TODO: change only one argument
                        nodeSetJVMArguments(genomeItem);
                    } else{
                        // set argument1 to random value (0-255)
                        nodeSetArgument(genomeItem, argIndex, GARandomInt(0, UCHAR_MAX));
                    }

                    genome.gene(offset + slot, genomeItem);
                }
            }
        }
    }
    return numOfMutations;
}

int GAGateCallbacks::crossover_perLayer(const GA1DArrayGenome<GENOME_ITEM_TYPE> &parent1, const GA1DArrayGenome<GENOME_ITEM_TYPE> &parent2,
                                    GA1DArrayGenome<GENOME_ITEM_TYPE> *offspring1, GA1DArrayGenome<GENOME_ITEM_TYPE> *offspring2) {
    // take random layer and cut individuals in 2 parts horisontally
    int crossPoint = GARandomInt(1,pGlobals->settings->gateCircuit.numLayers) * 2;
    for (int layer = 0; layer < 2 * pGlobals->settings->gateCircuit.numLayers; layer++) {
        int offset = layer * pGlobals->settings->gateCircuit.genomeWidth;
        if (layer <= crossPoint) {
            for (int i = 0; i < pGlobals->settings->gateCircuit.genomeWidth; i++) {
                if (offspring1 != NULL) offspring1->gene(offset + i, parent1.gene(offset + i));
                if (offspring2 != NULL) offspring2->gene(offset + i, parent2.gene(offset + i));
            }
        } else {
            for (int i = 0; i < pGlobals->settings->gateCircuit.genomeWidth; i++) {
                if (offspring1 != NULL) offspring1->gene(offset + i, parent2.gene(offset + i));
                if (offspring2 != NULL) offspring2->gene(offset + i, parent1.gene(offset + i));
            }
        }
    }
    // return number of offsprings
    return (offspring1 != NULL ? 1 : 0) + (offspring2 != NULL ? 1 : 0);
}
int GAGateCallbacks::crossover_perColumn(const GA1DArrayGenome<GENOME_ITEM_TYPE> &parent1, const GA1DArrayGenome<GENOME_ITEM_TYPE> &parent2,
                                     GA1DArrayGenome<GENOME_ITEM_TYPE> *offspring1, GA1DArrayGenome<GENOME_ITEM_TYPE> *offspring2) {
    // take random point in layer size and cut individuals in 2 parts verically
    int crossPoint = GARandomInt(1,pGlobals->settings->gateCircuit.sizeLayer);
    for (int layer = 0; layer < 2 * pGlobals->settings->gateCircuit.numLayers; layer++) {
        int offset = layer * pGlobals->settings->gateCircuit.genomeWidth;
        for (int i = 0; i < pGlobals->settings->gateCircuit.genomeWidth; i++) {
            if ( i < crossPoint) {
                if (offspring1 != NULL) offspring1->gene(offset + i, parent1.gene(offset + i));
                if (offspring2 != NULL) offspring2->gene(offset + i, parent2.gene(offset + i));
            } else {
                if (offspring1 != NULL) offspring1->gene(offset + i, parent2.gene(offset + i));
                if (offspring2 != NULL) offspring2->gene(offset + i, parent1.gene(offset + i));
            }
        }
    }
    // return number of offsprings
    return (offspring1 != NULL ? 1 : 0) + (offspring2 != NULL ? 1 : 0);
}

GENOME_ITEM_TYPE GAGateCallbacks::changeBit(GENOME_ITEM_TYPE genomeValue, int width) {
    return genomeValue ^ (1 << GARandomInt(0, width-1));
}
