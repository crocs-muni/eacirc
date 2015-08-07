#include "Converter.h"

#include "EACglobals.h"
#include "circuit/gate/GateCommonFunctions.h"


Converter::Converter()
{
    spec_.inSize = pGlobals->settings->testVectors.inputLength;
    spec_.outSize = pGlobals->settings->testVectors.outputLength;
    spec_.layerNum = pGlobals->settings->gateCircuit.numLayers;
    spec_.layerSize = pGlobals->settings->gateCircuit.sizeLayer;
}

void Converter::convert( const GaCirc& orig, Node* node ) const
{
    const int numConnectors = pGlobals->settings->gateCircuit.numConnectors;
    const int sizeInputLayer = pGlobals->settings->gateCircuit.sizeInputLayer;
    const int genomeWidth = pGlobals->settings->gateCircuit.genomeWidth;
    const int sizeOutputLayer = pGlobals->settings->gateCircuit.sizeOutputLayer;

    // convert first layer
    for (int i = 0; i < spec_.layerSize; ++i) {
        GENOME_ITEM_TYPE connMask = relativeToAbsoluteConnectorMask( orig.gene( i ), i, sizeInputLayer, sizeInputLayer );
        GENOME_ITEM_TYPE funcMask = orig.gene( genomeWidth + i );
        
        node->func = Func( nodeGetFunction( funcMask ) );
        for (int k = 0; k < Node::argvSize; ++k) node->argv[k] = nodeGetArgument( funcMask, k + 1 );
        node->conns = GenomeItem_t( connMask );
        
        ++node;
    }
    // convert middle layers
    for (int j = 1; j < spec_.layerNum - 1; ++j) {
        unsigned int offsetConns = 2 * j * genomeWidth;
        unsigned int offsetFuncs = offsetConns + genomeWidth;

        for (int i = 0; i < spec_.layerSize; ++i) {
            GENOME_ITEM_TYPE connMask = relativeToAbsoluteConnectorMask( orig.gene( offsetConns + i ), i, spec_.layerSize, numConnectors );
            GENOME_ITEM_TYPE funcMask = orig.gene( offsetFuncs + i );
            
            node->func = Func( nodeGetFunction( funcMask ) );
            for (int k = 0; k < Node::argvSize; ++k) node->argv[k] = nodeGetArgument( funcMask, k + 1 );
            node->conns = GenomeItem_t( connMask );

            ++node;
        }
    }
    // convert last layer
    unsigned int offsetConns = 2 * (spec_.layerNum - 1) * genomeWidth;
    unsigned int offsetFuncs = offsetConns + genomeWidth;
    for (int i = 0; i < sizeOutputLayer; ++i) {
        GENOME_ITEM_TYPE connMask = relativeToAbsoluteConnectorMask( orig.gene( offsetConns + i ), i % spec_.layerSize, spec_.layerSize, spec_.layerSize );
        GENOME_ITEM_TYPE funcMask = orig.gene( offsetFuncs + i );

        node->func = Func( nodeGetFunction( funcMask ) );
        for (int k = 0; k < Node::argvSize; ++k) node->argv[k] = nodeGetArgument( funcMask, k + 1 );
        node->conns = GenomeItem_t( connMask );

       ++node;
    }
}