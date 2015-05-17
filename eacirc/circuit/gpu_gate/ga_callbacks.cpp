#include "ga_callbacks.h"

#include "gate_helper.h"
#include "gpu_task.h"
#include "EACglobals.h"
#include "evaluators/IEvaluator.h"


float ga_callbacks::evaluator(GAGenome& genome)
{
    static const size_t vec_count = pGlobals->settings->testVectors.setSize;

    static gpu_task::circuit_type circuit;
    circuit.in_size = pGlobals->settings->testVectors.inputLength;
    circuit.out_size = pGlobals->settings->testVectors.outputLength;

    circuit.layer_size = pGlobals->settings->gateCircuit.sizeLayer;
    circuit.layer_num = pGlobals->settings->gateCircuit.numLayers;

    circuit.genome_width = pGlobals->settings->gateCircuit.genomeWidth;

    const size_t block_size = pGlobals->settings->cuda.block_size;

    static gpu_task task(circuit, vec_count, block_size);
    static gate_helper<GENOME_ITEM_TYPE, gpu_task::genome_item_t> helper;


    if (pGlobals->testVectors.newSet) {
        task.update_inputs(const_cast<const byte**>(pGlobals->testVectors.inputs), vec_count);
        pGlobals->testVectors.newSet = false;
    }

    helper.transform((const GAGenome*)&genome);

    task.update_circuit(helper.get_nodes());
    task.run();
    byte* outs = task.receive_outputs();



    pGlobals->evaluator->resetEvaluator();
    for ( size_t i = 0; i < vec_count; i++ ) {
        auto out_offset = outs + (i * circuit.out_size);
        pGlobals->evaluator->evaluateCircuit( out_offset, pGlobals->testVectors.outputs[i] );
    }
    return pGlobals->evaluator->getFitness();
}
