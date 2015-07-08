#pragma once

#include "Byte.h"
#include <cuda_runtime.h>
#include <limits>


template <typename T>
class GateCircuit
{
public:
    using ValueType = T;

    static const int maxConnIndex = std::numeric_limits<ValueType>::digits;
public:
    class Node {
    public:
        __device__ __host__ __forceinline__ Node() = default;

        __device__ __forceinline__ Node(ValueType funcMask, ValueType connMask) :
            funcMask(funcMask),
            connMask(connMask)
        {}

    public:
        __device__ __forceinline__ Byte getFunc() const
        {
            return funcMask & 0xff;
        }


        __device__ __forceinline__ Byte getArg(const int i) const
        {
            return static_cast<Byte>(funcMask >> ((4 - i) * std::numeric_limits<Byte>::digits)) & 0xff;
        }


        __device__ int extractNextConnector(int i) const
        {
            if (i == -2)
                return -2;
            ++i;
            while ((connMask & (static_cast<ValueType>(1) << i)) == 0 && i < maxConnIndex)
                ++i;
            return (i < maxConnIndex) ? i : -2;
        }


        __host__ void setFunc(ValueType funcMask) { this->funcMask = funcMask; }
        __host__ void setConn(ValueType connMask) { this->connMask = connMask; }

        __device__ ValueType getFuncMask() const { return funcMask; }
        __device__ ValueType getConnMask() const { return connMask; }

    private:
        ValueType funcMask;
        ValueType connMask;
    };

public:
    size_t inSize;
    size_t outSize;

    size_t layerSize;
    size_t layerNum;

    size_t genomeWidth;
    const Node* data;
};
