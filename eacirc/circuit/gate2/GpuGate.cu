#ifdef CUDA // compile if CUDA support is on

#include "GpuGate.h"
#include "cuda/cudaSafeCall.h"
#include "Interpreter.h"
#include <cmath>
#include <cuda_runtime_api.h>


// NOTE: change the maximum number of nodes if it is not enough
// BEWARE: the size of the device constant memory is limited (see CUDA documentation)
__constant__ __align__(128) GpuGate::Node devNodes[500];


__global__ void kernel( const GpuGate::Spec specs, const Byte* ins, Byte* outs, const int vecNum, const int bankSize )
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id < vecNum ) {
        // get offset to inputs and outputs
        const Byte* in = ins + (id * specs.inSize);
        Byte* out = outs + (id * specs.outSize);

        // allocate memory for execution layers
        extern __shared__ Byte memory[];
        Byte* layers = memory + ((2 * specs.inSize + bankSize) * threadIdx.x);

        // execute
        Interpreter interpreter( &specs, layers );
        interpreter.execute( devNodes, in, out );
    }
}


GpuGate::GpuGate( const Spec& specs, const int vecNum, const int blockSize ) :
    devIns_( nullptr ),
    devOuts_( nullptr ),
    spec_( specs ),
    vecNum_( vecNum ),
    blockSize_( blockSize )
{
    cudaSafeCall( cudaSetDevice( 0 ) );

    cudaSafeCall( cudaMalloc( reinterpret_cast<void**>(&devIns_), spec_.inSize * vecNum_ ) );
    cudaSafeCall( cudaMalloc( reinterpret_cast<void**>(&devOuts_), spec_.outSize * vecNum_ ) );

    cudaSharedMemConfig config;
    cudaSafeCall( cudaDeviceGetSharedMemConfig( &config ) );
    bankSize_ = (config == cudaSharedMemBankSizeEightByte) ? 8 : 4;
}

GpuGate::~GpuGate()
{
    cudaSafeCall( cudaFree( devIns_ ) );
    cudaSafeCall( cudaFree( devOuts_ ) );
}


void GpuGate::run( const Node* nodes )
{
    // copy circuit to device
    const int nodeNum = (spec_.layerNum - 1) * spec_.layerSize + spec_.outSize;
    cudaSafeCall( cudaMemcpyToSymbol( devNodes, nodes, nodeNum * sizeof( Node ) ) );

    // compute needed resources
    const int gridSize = int( std::ceil( float( vecNum_ ) / blockSize_ ) );
    const int memSize = blockSize_ * (2 * spec_.inSize + bankSize_);

    // launch on device
    kernel<<< gridSize, blockSize_, memSize >>>(spec_, devIns_, devOuts_, vecNum_, bankSize_);
}

void GpuGate::deployInputs( const TestVectors& ins )
{
    cudaSafeCall( cudaMemcpy( devIns_, ins.data(), spec_.inSize * vecNum_, cudaMemcpyHostToDevice ) );
}

void GpuGate::fetchOutputs( TestVectors& outs ) const
{
    cudaSafeCall( cudaMemcpy( outs.data(), devOuts_, spec_.outSize * vecNum_, cudaMemcpyDeviceToHost ) );
}

#endif // CUDA