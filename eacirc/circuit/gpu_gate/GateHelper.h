#pragma once

#include "GateCircuit.h"
#include "EACglobals.h"
#include "circuit/gate/GateCommonFunctions.h"
#include <GA1DArrayGenome.h>
#include <cuda_runtime_api.h>


template <typename S, typename D>
class GateHelper
{
public:
    using GaCircuit = GA1DArrayGenome<S>;

    using CircuitType = GateCircuit<D>;
    using CircuitNode = typename CircuitType::Node;
public:
    GateHelper() :
        hostNodes(nullptr)
    {
        const size_t layerNum = pGlobals->settings->gateCircuit.numLayers;
        const size_t genomeWidth = pGlobals->settings->gateCircuit.genomeWidth;

        cudaMallocHost(&hostNodes, layerNum * genomeWidth * sizeof(CircuitNode));
    }

    ~GateHelper()
    {
        cudaFreeHost(hostNodes);
    }

public:
    CircuitNode* getNodes() const { return hostNodes; }


    void transform(const GAGenome* src)
    {
        const GaCircuit* srcCircuit = dynamic_cast<const GaCircuit*>(src);

        const size_t outSize = pGlobals->settings->testVectors.outputLength;
        const size_t layerSize = pGlobals->settings->gateCircuit.sizeLayer;
        const size_t layerNum = pGlobals->settings->gateCircuit.numLayers;
        const size_t genomeWidth = pGlobals->settings->gateCircuit.genomeWidth;
        const size_t sizeLayer = pGlobals->settings->gateCircuit.sizeLayer;
        const size_t sizeInputLayer = pGlobals->settings->gateCircuit.sizeInputLayer;
        const size_t numConnectors = pGlobals->settings->gateCircuit.numConnectors;

        for (size_t layer = 0; layer < layerNum; layer++) {
            size_t offsetConns = 2 * layer * genomeWidth;
            size_t offsetFuncs = offsetConns + genomeWidth;

            size_t numSlots = (layer != layerNum - 1) ? layerSize : outSize;

            for (size_t i = 0; i < numSlots; i++) {
                D funcMask = srcCircuit->gene(static_cast<unsigned int>(offsetFuncs + i));
                D connMask = srcCircuit->gene(static_cast<unsigned int>(offsetConns + i));

                if (layer == 0) {
                    connMask = relativeToAbsoluteConnectorMask(connMask, static_cast<unsigned int>(i), static_cast<unsigned int>(sizeInputLayer), static_cast<unsigned int>(sizeInputLayer));
                } else if (layer == layerNum - 1) {
                    connMask = relativeToAbsoluteConnectorMask(connMask, static_cast<unsigned int>(i % sizeLayer), static_cast<unsigned int>(sizeLayer), static_cast<unsigned int>(sizeLayer));
                } else {
                    connMask = relativeToAbsoluteConnectorMask(connMask, static_cast<unsigned int>(i), static_cast<unsigned int>(sizeLayer), static_cast<unsigned int>(numConnectors));
                }

                hostNodes[(layer * genomeWidth) + i].setFunc(funcMask);
                hostNodes[(layer * genomeWidth) + i].setConn(connMask);
            }
        }
    }

private:
    CircuitNode* hostNodes;
};
