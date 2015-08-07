#pragma once

#include <stdexcept>
#include <cuda_runtime.h>


// TODO: create class for fatal exceptions and handle it properly
__forceinline__ void cudaSafeCall( const cudaError_t err )
{
    if ( err != cudaSuccess )
        throw std::runtime_error( cudaGetErrorString( err ) );
}