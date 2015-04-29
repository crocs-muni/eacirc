#pragma once

#include "byte.h"
#include "gate_circuit.h"
#include "EACconstants.h"
#include <cuda_runtime.h>


template <typename T>
class gate_interpreter
{
public:
    using circuit_t = gate_circuit<T>;
    using circuit_node = typename circuit_t::node;
public:
    __device__ __forceinline__ gate_interpreter(byte* layers, const circuit_t* circuit) :
        _circuit(circuit),
        _layer_out(layers),
        _layer_in(layers + circuit->in_size),
        _original_in(nullptr)
    {}

public:
    __device__ __forceinline__ bool execute(const byte* in, byte* out)
    {
        for (int i = 0; i < _circuit->in_size; ++i)
            _layer_in[i] = in[i];
        _original_in = in;


        const circuit_node* circuit_offset = _circuit->data;

        for (int j = 0; j < _circuit->layer_num - 1; ++j) {
            for (int i = 0; i < _circuit->layer_size; ++i) {
                if ( !execute_node(circuit_offset[i], _layer_out[i]) )
                    return false;
            }

            swap_layers();
            circuit_offset += _circuit->genome_width;
        }
        for (int i = 0; i < _circuit->out_size; ++i) {
            if ( !execute_node(circuit_offset[i], out[i]) )
                return false;
        }

        return true;
    }


    __device__ static byte get_func_neutral_value(const byte func)
    {
        switch (func) {
        case FNC_AND:
        case FNC_NAND:
            return std::numeric_limits< byte >::max();
        default:
            return 0;
        }
    }

protected:
    __device__ __forceinline__ void swap_layers()
    {
        byte* temp = _layer_in;
        _layer_in = _layer_out;
        _layer_out = temp;
    }


    __device__ bool execute_node(const circuit_node& node, byte& out) const;

private:
    byte* _layer_in;
    byte* _layer_out;
    const byte* _original_in;
    const circuit_t* _circuit;
};



template <typename T>
__device__ bool gate_interpreter<T>::execute_node(const circuit_node& node, byte& out) const
{
    const byte func = node.get_func();
    const byte arg1 = node.get_arg(1);

    byte result = get_func_neutral_value(func);

    int conn1 = -1;
    int conn2 = -1;


    switch (func) {
    case FNC_NOP:
        if ((conn1 = node.extract_next_connector(conn1)) >= 0) result = _layer_in[conn1];
        break;
    case FNC_CONS:
        result = arg1;
        break;
    case FNC_AND:
        while ((conn1 = node.extract_next_connector(conn1)) >= 0) result &= _layer_in[conn1];
        break;
    case FNC_NAND:
        while ((conn1 = node.extract_next_connector(conn1)) >= 0) result &= _layer_in[conn1];
        result = ~result;
        break;
    case FNC_OR:
        while ((conn1 = node.extract_next_connector(conn1)) >= 0) result |= _layer_in[conn1];
        break;
    case FNC_XOR:
        while ((conn1 = node.extract_next_connector(conn1)) >= 0) result ^= _layer_in[conn1];
        break;
    case FNC_NOR:
        while ((conn1 = node.extract_next_connector(conn1)) >= 0) result |= _layer_in[conn1];
        result = ~result;
        break;
    case FNC_NOT:
        if ((conn1 = node.extract_next_connector(conn1)) >= 0) result = ~(_layer_in[conn1]);
        break;
    case FNC_SHIL:
        if ((conn1 = node.extract_next_connector(conn1)) >= 0) result = _layer_in[conn1] << (arg1 % std::numeric_limits< byte >::digits);
        break;
    case FNC_SHIR:
        if ((conn1 = node.extract_next_connector(conn1)) >= 0) result = _layer_in[conn1] >> (arg1 % std::numeric_limits< byte >::digits);
        break;
    case FNC_ROTL:
        if ((conn1 = node.extract_next_connector(conn1)) >= 0) {
            if (arg1 % std::numeric_limits< byte >::digits != 0) {
                result = (_layer_in[conn1] << (arg1 % std::numeric_limits< byte >::digits))
                       | (_layer_in[conn1] >> (std::numeric_limits< byte >::digits - arg1 % std::numeric_limits< byte >::digits));
            }
        }
        break;
    case FNC_ROTR:
        if ((conn1 = node.extract_next_connector(conn1)) >= 0) {
            if (arg1 % std::numeric_limits< byte >::digits != 0) {
                result = (_layer_in[conn1] >> (arg1 % std::numeric_limits< byte >::digits))
                       | (_layer_in[conn1] << (std::numeric_limits< byte >::digits - arg1 % std::numeric_limits< byte >::digits));
            }
        }
        break;
    case FNC_EQ:
        if ((conn2 = node.extract_next_connector((conn1 = node.extract_next_connector(-1)))) >= 0) {
            if (_layer_in[conn1] == _layer_in[conn2]) result = std::numeric_limits< byte >::max();
        }
        break;
    case FNC_LT:
        if ((conn2 = node.extract_next_connector((conn1 = node.extract_next_connector(-1)))) >= 0) {
            if (_layer_in[conn1] < _layer_in[conn2]) result = std::numeric_limits< byte >::max();
        }
        break;
    case FNC_GT:
        if ((conn2 = node.extract_next_connector((conn1 = node.extract_next_connector(-1)))) >= 0) {
            if (_layer_in[conn1] > _layer_in[conn2]) result = std::numeric_limits< byte >::max();
        }
        break;
    case FNC_LEQ:
        if ((conn2 = node.extract_next_connector((conn1 = node.extract_next_connector(-1)))) >= 0) {
            if (_layer_in[conn1] <= _layer_in[conn2]) result = std::numeric_limits< byte >::max();
        }
        break;
    case FNC_GEQ:
        if ((conn2 = node.extract_next_connector((conn1 = node.extract_next_connector(-1)))) >= 0) {
            if (_layer_in[conn1] >= _layer_in[conn2]) result = std::numeric_limits< byte >::max();
        }
        break;
    case FNC_BSLC:
        if ((conn1 = node.extract_next_connector(-1)) >= 0) result = _layer_in[conn1] & arg1;
        break;
    case FNC_READ:
        result = _original_in[arg1 % _circuit->in_size];
        break;
    default:
        return false;
    }

    out = result;

    return true;
}
