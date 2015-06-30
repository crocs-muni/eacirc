#include "GpuGate.h"
#include "EACglobals.h"
#include "Logger.h"
#include <stdexcept>


void GpuGate::check_settings() const
{
    try {
        if (pGlobals->settings->gateCircuit.useMemory)
            throw std::invalid_argument("Bad settings! Cuda backend cannot have useMemory enabled.");
        if (pGlobals->settings->gateCircuit.allowedFunctions[FNC_JVM])
            throw std::invalid_argument("Bad settings! Cuda backend cannot have FNC_JVM function enabled.");
    }
    catch (std::exception& e) {
        mainLogger.out(LOGGER_ERROR) << e.what() << std::endl;
        std::exit(1);
    }
}