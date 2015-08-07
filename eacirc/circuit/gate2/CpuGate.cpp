#include "CpuGate.h"
#include "Interpreter.h"


void CpuGate::run( const Node* circ, const TestVectors& ins, TestVectors& outs )
{
    Interpreter interpreter( &spec_, layers_.get() );
    for ( int i = 0; i < ins.num(); ++i ) {
        interpreter.execute( circ, ins[i], outs[i] );
    }
}