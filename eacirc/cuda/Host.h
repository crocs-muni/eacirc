#pragma once

#include "cudaSafeCall.h"
#include <memory>
#include <cuda_runtime.h>

namespace cuda {
    struct Host
    {
        static void* malloc( size_t size )
        {
            void* p = nullptr;
            cudaSafeCall( cudaMallocHost( &p, size ) );
            return p;
        }

        static void free( void* ptr )
        {
            cudaSafeCall( cudaFreeHost( ptr ) );
        }
        
        // NOTE: only frees memmory, no destructor is called
        template <class T>
        struct default_delete
        {
            void operator() ( T* ptr ) { Host::free( ptr ); }
        };

        template <class T>
        struct default_delete<T[]>
        {
            void operator() ( T* ptr ) { Host::free( ptr ); }
        };
        
        template <class T>
        using unique_ptr = std::unique_ptr<T, Host::default_delete<T>>;
        
        // NOTE: only allocates memmory, no constructor is called
        template <class T>
        static std::enable_if_t<std::is_array<T>::value && std::extent<T>::value == 0, unique_ptr<T>>
            make_unique( size_t size )
        {
            using E = std::remove_extent_t<T>;
            return unique_ptr<T>( reinterpret_cast<E*>(Host::malloc( size * sizeof( E ) )) );
        }
    };
}