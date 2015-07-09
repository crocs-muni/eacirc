#pragma once

#include "Byte.h"
#include "GateCircuit.h"
#include "EACconstants.h"
#include <cuda_runtime.h>


template <typename T>
class GateInterpreter
{
public:
    using CircuitType = GateCircuit<T>;
    using CircuitNode = typename CircuitType::Node;

    // numeric limit of byte, nvcc on windows in vs2013 doesnt know std::numeric_limits, linux is fine
    static constexpr Byte byteMaxLimit = 255;
public:
    __device__ __forceinline__ GateInterpreter(Byte* layers, const CircuitType* circuit) :
        circuit(circuit),
        layerOut(layers),
        layerIn(layers + circuit->inSize),
        originalIn(nullptr)
    {}

public:
    __device__ __forceinline__ bool execute(const Byte* in, Byte* out)
    {
        for (int i = 0; i < circuit->inSize; ++i)
            layerIn[i] = in[i];
        originalIn = in;

        const CircuitNode* circuitOffset = circuit->data;

        for (int j = 0; j < circuit->layerNum - 1; ++j) {
            for (int i = 0; i < circuit->layerSize; ++i) {
                if ( !executeNode(circuitOffset[i], layerOut[i]) )
                    return false;
            }

            swapLayers();
            circuitOffset += circuit->genomeWidth;
        }
        for (int i = 0; i < circuit->outSize; ++i) {
            if ( !executeNode(circuitOffset[i], out[i]) )
                return false;
        }
        return true;
    }


    __device__ static Byte getFuncNeutralValue(const Byte func)
    {
        switch (func) {
        case FNC_AND:
        case FNC_NAND:
            return byteMaxLimit;
        default:
            return 0;
        }
    }

protected:
    __device__ __forceinline__ void swapLayers()
    {
        Byte* temp = layerIn;
        layerIn = layerOut;
        layerOut = temp;
    }


    __device__ bool executeNode(const CircuitNode& node, Byte& out) const;

private:
    Byte* layerIn;
    Byte* layerOut;
    const Byte* originalIn;
    const CircuitType* circuit;
};



template <typename T>
__device__ bool GateInterpreter<T>::executeNode(const CircuitNode& node, Byte& out) const
{
    const Byte func = node.getFunc();
    const Byte arg1 = node.getArg(1);

    Byte result = getFuncNeutralValue(func);

    int conn1 = -1;
    int conn2 = -1;


    switch (func) {
    case FNC_NOP:
        if ((conn1 = node.extractNextConnector(conn1)) >= 0) result = layerIn[conn1];
        break;
    case FNC_CONS:
        result = arg1;
        break;
    case FNC_AND:
        while ((conn1 = node.extractNextConnector(conn1)) >= 0) result &= layerIn[conn1];
        break;
    case FNC_NAND:
        while ((conn1 = node.extractNextConnector(conn1)) >= 0) result &= layerIn[conn1];
        result = ~result;
        break;
    case FNC_OR:
        while ((conn1 = node.extractNextConnector(conn1)) >= 0) result |= layerIn[conn1];
        break;
    case FNC_XOR:
        while ((conn1 = node.extractNextConnector(conn1)) >= 0) result ^= layerIn[conn1];
        break;
    case FNC_NOR:
        while ((conn1 = node.extractNextConnector(conn1)) >= 0) result |= layerIn[conn1];
        result = ~result;
        break;
    case FNC_NOT:
        if ((conn1 = node.extractNextConnector(conn1)) >= 0) result = ~(layerIn[conn1]);
        break;
    case FNC_SHIL:
        if ((conn1 = node.extractNextConnector(conn1)) >= 0) result = layerIn[conn1] << (arg1 % std::numeric_limits< Byte >::digits);
        break;
    case FNC_SHIR:
        if ((conn1 = node.extractNextConnector(conn1)) >= 0) result = layerIn[conn1] >> (arg1 % std::numeric_limits< Byte >::digits);
        break;
    case FNC_ROTL:
        if ((conn1 = node.extractNextConnector(conn1)) >= 0) {
            if (arg1 % std::numeric_limits< Byte >::digits != 0) {
                result = (layerIn[conn1] << (arg1 % std::numeric_limits< Byte >::digits))
                       | (layerIn[conn1] >> (std::numeric_limits< Byte >::digits - arg1 % std::numeric_limits< Byte >::digits));
            }
        }
        break;
    case FNC_ROTR:
        if ((conn1 = node.extractNextConnector(conn1)) >= 0) {
            if (arg1 % std::numeric_limits< Byte >::digits != 0) {
                result = (layerIn[conn1] >> (arg1 % std::numeric_limits< Byte >::digits))
                       | (layerIn[conn1] << (std::numeric_limits< Byte >::digits - arg1 % std::numeric_limits< Byte >::digits));
            }
        }
        break;
    case FNC_EQ:
        if ((conn2 = node.extractNextConnector((conn1 = node.extractNextConnector(-1)))) >= 0) {
            if (layerIn[conn1] == layerIn[conn2]) result = byteMaxLimit;
        }
        break;
    case FNC_LT:
        if ((conn2 = node.extractNextConnector((conn1 = node.extractNextConnector(-1)))) >= 0) {
            if (layerIn[conn1] < layerIn[conn2]) result = byteMaxLimit;
        }
        break;
    case FNC_GT:
        if ((conn2 = node.extractNextConnector((conn1 = node.extractNextConnector(-1)))) >= 0) {
            if (layerIn[conn1] > layerIn[conn2]) result = byteMaxLimit;
        }
        break;
    case FNC_LEQ:
        if ((conn2 = node.extractNextConnector((conn1 = node.extractNextConnector(-1)))) >= 0) {
            if (layerIn[conn1] <= layerIn[conn2]) result = byteMaxLimit;
        }
        break;
    case FNC_GEQ:
        if ((conn2 = node.extractNextConnector((conn1 = node.extractNextConnector(-1)))) >= 0) {
            if (layerIn[conn1] >= layerIn[conn2]) result = byteMaxLimit;
        }
        break;
    case FNC_BSLC:
        if ((conn1 = node.extractNextConnector(-1)) >= 0) result = layerIn[conn1] & arg1;
        break;
    case FNC_READ:
        result = originalIn[arg1 % circuit->inSize];
        break;
    default:
        return false;
    }

    out = result;
    return true;
}
