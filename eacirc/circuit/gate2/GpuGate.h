#pragma once
#ifdef CUDA // compile if CUDA support is on

#include "TestVectors.h"
#include "Circuit.h"


class GpuGate
{
public:
    using Spec = Circuit::Spec;
    using Node = Circuit::Node;
public:
    GpuGate( const Spec& specs, const int vecNum, const int blockSize );
    ~GpuGate();
public:
    void run( const Node* nodes );
    void deployInputs( const TestVectors& ins );
    void fetchOutputs( TestVectors& outs ) const;

    void run( const Node* nodes, const TestVectors& ins, TestVectors& outs )
    {
        deployInputs( ins );
        run( nodes );
        fetchOutputs( outs );
    }
private:
    Byte* devIns_;
    Byte* devOuts_;
    Spec spec_;

    int vecNum_;
    int bankSize_;
    int blockSize_;

};

#endif // CUDA