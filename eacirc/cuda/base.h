#pragma once

#ifdef CUDA
    #include <cuda_runtime.h>

    #define DEVICE __device__
    #define HOST __host__
    #define FORCEINLINE __forceinline__
#else
    #define DEVICE
    #define HOST
    #define FORCEINLINE
#endif