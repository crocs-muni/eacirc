#pragma once

#include "Circuit.h"
#include "EACglobals.h"
#include <algorithm>
#include <GA1DArrayGenome.h>


class Converter
{
    using Spec = Circuit::Spec;
    using Node = Circuit::Node;
    using Func = Circuit::Func;
    using GaCirc = GA1DArrayGenome<GENOME_ITEM_TYPE>;
public:
    Converter();

    const Spec& spec() const { return spec_; }
    int nodeNum() const { return (spec_.layerNum - 1) * spec_.layerSize + spec_.outSize; }

    void convert( const GaCirc& orig, Node* node ) const;
    void convert( const GAGenome& genome, Node* node ) const { convert( dynamic_cast<const GaCirc&>(genome), node ); }
private:
    Spec spec_;
};