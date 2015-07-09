#include "GaCallbacks.h"

#include "GateHelper.h"
#include "GpuTask.h"
#include "EACglobals.h"
#include "evaluators/IEvaluator.h"


float GaCallbacks::evaluator(GAGenome& genome)
{
    static const size_t vecCount = pGlobals->settings->testVectors.setSize;

    static GpuTask::CircuitType circuit;
    circuit.inSize = pGlobals->settings->testVectors.inputLength;
    circuit.outSize = pGlobals->settings->testVectors.outputLength;

    circuit.layerSize = pGlobals->settings->gateCircuit.sizeLayer;
    circuit.layerNum = pGlobals->settings->gateCircuit.numLayers;

    circuit.genomeWidth = pGlobals->settings->gateCircuit.genomeWidth;

    const size_t blockSize = pGlobals->settings->cuda.block_size;

    static GpuTask task(circuit, vecCount, blockSize);
    static GateHelper<GENOME_ITEM_TYPE, GpuTask::GenomeItem> helper;

    //deploy to GPU and execute
    if (pGlobals->testVectors.newSet) {
        task.updateInputs(const_cast<const Byte**>(pGlobals->testVectors.inputs), vecCount);
        pGlobals->testVectors.newSet = false;
    }

    helper.transform((const GAGenome*)&genome);

    task.updateCircuit(helper.getNodes());
    task.run();
    Byte* outs = task.receiveOutputs();

    //evaluate
    pGlobals->evaluator->resetEvaluator();
    for ( size_t i = 0; i < vecCount; i++ ) {
        auto outOffset = outs + (i * circuit.outSize);
        pGlobals->evaluator->evaluateCircuit( outOffset, pGlobals->testVectors.outputs[i] );
    }
    return pGlobals->evaluator->getFitness();
}
