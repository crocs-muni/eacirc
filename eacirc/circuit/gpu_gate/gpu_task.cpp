#include "gpu_task.h"

#include "gate_interpreter.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <cmath>



gpu_task::gpu_task(const circuit_type& circuit, const size_t vec_count, const size_t block_size) :
    _vec_count(static_cast<size_t>(std::ceil(float(vec_count) / block_size) * block_size)),
    _bank_size(0),
    _block_size(block_size),
    _dev_ins(nullptr),
    _dev_outs(nullptr),
    _host_ins(nullptr),
    _host_outs(nullptr),
    _circuit(circuit)
{
    cudaSetDevice(0);

    cudaMallocHost(&_host_ins, _circuit.in_size * _vec_count);
    cudaMallocHost(&_host_outs, _circuit.out_size * _vec_count);

    cudaMalloc(&_dev_ins, _circuit.in_size * _vec_count);
    cudaMalloc(&_dev_outs, _circuit.out_size * _vec_count);

    cudaSharedMemConfig config;
    cudaDeviceGetSharedMemConfig(&config);
    _bank_size = (config == cudaSharedMemBankSizeEightByte) ? 8 : 4;
}


gpu_task::~gpu_task()
{
    cudaFree(_dev_ins);
    cudaFree( _dev_outs );

    cudaFreeHost(_host_ins);
    cudaFreeHost(_host_outs);
}



void gpu_task::update_inputs(const byte** ins, const size_t n)
{
    for ( size_t i = 0; i < n; ++i) {
        std::memcpy(_host_ins + (i * _circuit.in_size), ins[i], _circuit.in_size);
    }
    cudaMemcpy(_dev_ins, _host_ins, _circuit.in_size * n, cudaMemcpyHostToDevice);
}


byte* gpu_task::receive_outputs()
{
    cudaMemcpy(_host_outs, _dev_outs, _circuit.out_size * _vec_count, cudaMemcpyDeviceToHost);
    return _host_outs;
}
