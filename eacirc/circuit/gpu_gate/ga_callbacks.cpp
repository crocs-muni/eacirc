#include "ga_callbacks.h"

#include "gate_helper.h"
#include "gpu_task.h"
#include "EACglobals.h"
#include "evaluators/IEvaluator.h"

#include "circuit/gate/GateInterpreter.h"


float ga_callbacks::evaluator(GAGenome& genome)
{
    static const size_t vec_count = pGlobals->settings->testVectors.setSize;

    static gpu_task::circuit_type circuit;
    circuit.in_size = pGlobals->settings->testVectors.inputLength;
    circuit.out_size = pGlobals->settings->testVectors.outputLength;

    circuit.layer_size = pGlobals->settings->gateCircuit.sizeLayer;
    circuit.layer_num = pGlobals->settings->gateCircuit.numLayers;

    circuit.genome_width = pGlobals->settings->gateCircuit.genomeWidth;


    static gpu_task task(circuit, vec_count, 128);
    static gate_helper<GENOME_ITEM_TYPE, gpu_task::genome_item_t> helper;


    if (pGlobals->testVectors.newSet) {
        task.update_inputs(const_cast<const byte**>(pGlobals->testVectors.inputs), vec_count);
        pGlobals->testVectors.newSet = false;
    }


    helper.transform((const GAGenome*)&genome);
    task.update_circuit(helper.get_nodes());
    task.run();
    byte* outs = task.receive_outputs();


    GA1DArrayGenome<GENOME_ITEM_TYPE>  &genome2 = (GA1DArrayGenome<GENOME_ITEM_TYPE>&) genome;

    for (size_t testVector = 0; testVector < vec_count; testVector++) {
        GateInterpreter::executeCircuit(&genome2, pGlobals->testVectors.inputs[testVector], pGlobals->testVectors.circuitOutputs[testVector]);
    }


    for (size_t testVector = 0; testVector < vec_count; testVector++) {
        byte* out_offset = outs + (testVector * pGlobals->settings->testVectors.outputLength);
        for ( int i = 0; i < pGlobals->settings->testVectors.outputLength; ++i ) {
            if ( out_offset[i] !=  pGlobals->testVectors.circuitOutputs[testVector][i] ) {
                std::cout << std::endl << "mismatch: " << testVector << ": " << std::endl;
                //printf("%u %u\n", out_offset[i], pGlobals->testVectors.circuitOutputs[testVector][i]);
                exit(EXIT_FAILURE);
            }
        }
    }


    pGlobals->evaluator->resetEvaluator();
    for ( size_t i = 0; i < vec_count; i++ ) {
        auto out_offset = outs + (i * circuit.out_size);
        pGlobals->evaluator->evaluateCircuit( out_offset, pGlobals->testVectors.outputs[i] );
    }
    return pGlobals->evaluator->getFitness();
}
