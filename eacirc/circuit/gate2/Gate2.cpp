#include "Gate2.h"
#include "EACglobals.h"
#include "Logger.h"
#include <stdexcept>


std::string Gate2::shortDescription()
{
    return std::string( "Gate2 emulator " )
        + (pGlobals->settings->cuda.enabled ? std::string( "[CUDA]" ) : std::string( ("[CPU]") ));
}


int Gate2::loadCircuitConfiguration( TiXmlNode* pRoot )
{
    GateCircuit::loadCircuitConfiguration( pRoot );

    try {
        if (pGlobals->settings->gateCircuit.useMemory)
            throw std::invalid_argument( "Bad settings! Gate2 backend cannot have useMemory enabled." );
        if (pGlobals->settings->gateCircuit.allowedFunctions[FNC_JVM])
            throw std::invalid_argument( "Bad settings! Gate2 backend cannot have FNC_JVM function enabled." );
    }
    catch (std::exception& e) {
        mainLogger.out( LOGGER_ERROR ) << e.what() << std::endl;
        return STAT_CONFIG_INCORRECT;
    }
    return STAT_OK;
}