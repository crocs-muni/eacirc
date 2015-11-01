#include "GaGate.h"

#include "Converter.h"
#include "CpuGate.h"
#ifdef CUDA
    #include "GpuGate.h"
    #include "cuda/Host.h"
#endif
#include "evaluators/IEvaluator.h"


float GaGate::evaluator( GAGenome& genome )
{
    const int vecNum = pGlobals->settings->testVectors.setSize;
    static const Converter converter;
    
    // if copiled with CUDA support decide to use CUDA runner or CPU runner according to the settings
#ifdef CUDA
    if (pGlobals->settings->cuda.enabled) {
        static GpuGate runner( converter.spec(), vecNum, pGlobals->settings->cuda.block_size );
        static auto circ = cuda::Host::make_unique<Circuit::Node[]>( converter.nodeNum() );
        converter.convert( genome, circ.get() );
        runner.run( circ.get(), pGlobals->testVectors.inputs, pGlobals->testVectors.circuitOutputs );
    }
    else {
#endif
        static CpuGate runner( converter.spec() );
        static auto circ = std::make_unique<Circuit::Node[]>( converter.nodeNum() );
        converter.convert( genome, circ.get() );
        runner.run( circ.get(), pGlobals->testVectors.inputs, pGlobals->testVectors.circuitOutputs );
#ifdef CUDA
    }
#endif

    IEvaluator* evaluator = pGlobals->evaluator;
    evaluator->resetEvaluator();
    for (int i = 0; i < vecNum; ++i) {
        evaluator->evaluateCircuit( pGlobals->testVectors.circuitOutputs[i], pGlobals->testVectors.outputs[i] );
    }
    return evaluator->getFitness();
}
