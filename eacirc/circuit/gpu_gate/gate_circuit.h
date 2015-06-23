#pragma once

#include "byte.h"
#include <cuda_runtime.h>
#include <limits>


template <typename T>
class gate_circuit
{
public:
    using value_type = T;

    static const int max_conn_index = std::numeric_limits<value_type>::digits;
public:
    class node {
    public:
        __device__ __host__ __forceinline__ node() = default;

        __device__ __forceinline__ node(value_type func_mask, value_type conn_mask) :
            _func_mask(func_mask),
            _conn_mask(conn_mask)
        {}

    public:
        __device__ __forceinline__ byte get_func() const
        {
            return _func_mask & 0xff;
        }


        __device__ __forceinline__ byte get_arg(const int i) const
        {
            return static_cast<byte>(_func_mask >> ((4 - i) * std::numeric_limits<byte>::digits)) & 0xff;
        }


        __device__ int extract_next_connector(int i) const
        {
            if (i == -2)
                return -2;
            ++i;
            while ((_conn_mask & (static_cast<value_type>(1) << i)) == 0 && i < max_conn_index)
                ++i;
            return (i < max_conn_index) ? i : -2;
        }


        __host__ void set_func(value_type func_mask) { _func_mask = func_mask; }
        __host__ void set_conn(value_type conn_mask) { _conn_mask = conn_mask; }

        __device__ value_type get_func_mask() const { return _func_mask; }
        __device__ value_type get_conn_mask() const { return _conn_mask; }

    private:
        value_type _func_mask;
        value_type _conn_mask;
    };

public:
    size_t in_size;
    size_t out_size;

    size_t layer_size;
    size_t layer_num;

    size_t genome_width;
    const node* data;
};
