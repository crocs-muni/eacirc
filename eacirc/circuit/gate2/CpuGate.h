#pragma once

#include "Circuit.h"
#include "TestVectors.h"
#include <memory>


class CpuGate
{
public:
    using Spec = Circuit::Spec;
    using Node = Circuit::Node;
public:
    CpuGate( const Spec& specs )
        : spec_( specs )
        , layers_( std::make_unique<Byte[]>( 2 * specs.inSize ) )
    {}
public:
    void run( const Node* circ, const TestVectors& ins, TestVectors& outs );
private:
    Spec spec_;
    std::unique_ptr<Byte[]> layers_;
};