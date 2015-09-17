#pragma once

#include "Circuit.h"
#include "cuda/base.h"
#include <assert.h>

// TODO: remove this include, when start using enum as desribed in TODO a few lines below
#include "EACconstants.h"


class Interpreter
{
public:
    using Func = Circuit::Func;
    using Node = Circuit::Node;
    using Spec = Circuit::Spec;
    using Conn = Circuit::Conn;
public:
    DEVICE HOST FORCEINLINE Interpreter( const Spec* spec, Byte* layers )
        : spec_( spec )
        , in_( layers )
        , out_( in_ + spec->inSize )
    {}
public:
    DEVICE HOST FORCEINLINE void execute( const Node* nodes, const Byte* in, Byte* out )
    {
        for (int i = 0; i < spec_->inSize; ++i) {
            in_[i] = in[i];
        }

        for (int j = 0; j < spec_->layerNum - 1; ++j) {
            for (int i = 0; i < spec_->layerSize; ++i) {
                out_[i] = executeNode( *nodes, in );
                ++nodes;
            }
            swapLayers();
        }
        for (int i = 0; i < spec_->outSize; ++i) {
            out[i] = executeNode( *nodes, in );
            ++nodes;
        }
    }

protected:
    DEVICE HOST FORCEINLINE void swapLayers()
    {
        auto t = in_;
        in_ = out_;
        out_ = t;
    }

    DEVICE HOST Byte executeNode( const Node& node, const Byte* orig ) const
    {
        Byte result = 0;
        int i = -1;

        // TODO: make this a constexpr when all compilers will support it
        enum : int { digits = std::numeric_limits<decltype(result)>::digits };

        switch (node.func) {
        case FNC_NOP:
            if (node.extractConn( i ))
                result = in_[i];
            break;
        case FNC_CONS:
            result = node.argv[0];
            break;
        case FNC_AND:
            result = 0xff;
            while (node.extractConn( i ))
                result &= in_[i];
            break;
        case FNC_NAND:
            result = 0xff;
            while (node.extractConn( i ))
                result &= in_[i];
            result = ~result;
            break;
        case FNC_OR:
            while (node.extractConn( i ))
                result |= in_[i];
            break;
        case FNC_XOR:
            while (node.extractConn( i ))
                result ^= in_[i];
            break;
        case FNC_NOR:
            while (node.extractConn( i ))
                result |= in_[i];
            result = ~result;
            break;
        case FNC_NOT:
            if (node.extractConn( i ))
                result = ~(in_[i]);
            break;
        case FNC_SHIL:
            if (node.extractConn( i ))
                result = in_[i] << (node.argv[0] % digits);
            break;
        case FNC_SHIR:
            if (node.extractConn( i ))
                result = in_[i] >> (node.argv[0] % digits);
            break;
        case FNC_ROTL:
            if (node.extractConn( i ) && node.argv[0] % digits != 0)
                result = (in_[i] << (node.argv[0] % digits))
                | (in_[i] >> (digits - node.argv[0] % digits));
            break;
        case FNC_ROTR:
            if (node.extractConn( i ) && node.argv[0] % digits != 0)
                result = (in_[i] >> (node.argv[0] % digits))
                | (in_[i] << (digits - node.argv[0] % digits));
            break;
        case FNC_EQ:
            break;
        case FNC_LT:
            break;
        case FNC_GT:
            break;
        case FNC_LEQ:
            break;
        case FNC_GEQ:
            break;
        case FNC_BSLC:
            if (node.extractConn( i ))
                result = in_[i] & node.argv[0];
            break;
        case FNC_READ:
            result = orig[node.argv[0] % spec_->inSize];
            break;
        default:
            // fail if function is invalid
            assert( false );
        }
        return result;
    }
private:
    const Spec* spec_;
    Byte*  in_;
    Byte*  out_;
};