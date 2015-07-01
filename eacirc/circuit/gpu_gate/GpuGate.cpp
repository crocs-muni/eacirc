#include "GpuGate.h"
#include "EACglobals.h"
#include "Logger.h"
#include <stdexcept>


int GpuGate::loadCircuitConfiguration(TiXmlNode* pRoot)
{
    GateCircuit::loadCircuitConfiguration(pRoot);

    try {
        if (pGlobals->settings->gateCircuit.useMemory)
            throw std::invalid_argument("Bad settings! Cuda backend cannot have useMemory enabled.");
        if (pGlobals->settings->gateCircuit.allowedFunctions[FNC_JVM])
            throw std::invalid_argument("Bad settings! Cuda backend cannot have FNC_JVM function enabled.");
    }
    catch (std::exception& e) {
        mainLogger.out(LOGGER_ERROR) << e.what() << std::endl;
        return STAT_CONFIG_INCORRECT;
    }
    return STAT_OK;
}