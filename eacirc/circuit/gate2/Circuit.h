#pragma once

#include "Byte.h"
#include "GenomeItem_t.h"
#include "cuda/base.h"
#include <limits>


struct Circuit
{
    struct Spec
    {
        int inSize;
        int outSize;

        int layerSize;
        int layerNum;
    };

    // TODO: make the FUNC an enum as describd below (must corespond to the gate1 functions)
    //enum struct Func : Byte { NOP, CONS, AND, NAND, OR, XOR, NOR, NOT, SHIL, SHIR, ROTL, ROTR, EQ, LT, GT, LEQ, GEQ, BSLC, READ };
    using Func = Byte;
    using Conn = GenomeItem_t;

    struct Node
    {
        // TODO: make this a constexpr when all compilers supports it
        enum : int { argvSize = 3 };

        Byte argv[argvSize];
        Func func;
        Conn conns;
        
        DEVICE HOST FORCEINLINE bool extractConn( int& i ) const
        {
            // TODO: make this a constexpr when all compilers will support it
            enum : int { digits = std::numeric_limits<Conn>::digits };

            do { ++i; } while ( !((conns >> i) & Conn( 0x01 )) && i < digits );
            return i < digits ? true : false;
        }
    };
};