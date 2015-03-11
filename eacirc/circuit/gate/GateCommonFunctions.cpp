#include "GateCommonFunctions.h"

#define BOTTOM_BYTE 0xff

unsigned char nodeGetFunction(GENOME_ITEM_TYPE nodeValue) {
    return nodeValue & BOTTOM_BYTE;
}

unsigned char nodeGetArgument(GENOME_ITEM_TYPE nodeValue, int argumentNumber) {
    if (argumentNumber < 1 || argumentNumber > 4 ) {
        mainLogger.out(LOGGER_ERROR) << "Getting invalid argument: " << argumentNumber << "." << endl;
        return 0;
    }
    int shift = (4 - argumentNumber) * BITS_IN_UCHAR;
    return ((nodeValue & ((GENOME_ITEM_TYPE) BOTTOM_BYTE << shift)) >> shift) & BOTTOM_BYTE;
}

void nodeSetFunction(GENOME_ITEM_TYPE& nodeValue, unsigned char function) {
    nodeValue = nodeValue & (~BOTTOM_BYTE);
    nodeValue |= function;
}

void nodeSetArgument(GENOME_ITEM_TYPE& nodeValue, int argumentNumber, unsigned char argumentValue) {
    if (argumentNumber < 1 || argumentNumber > 4 ) {
        mainLogger.out(LOGGER_ERROR) << "Setting invalid argument: " << argumentNumber << "." << endl;
    }
    int shift = (4 - argumentNumber) * BITS_IN_UCHAR;
    nodeValue = nodeValue & (~((GENOME_ITEM_TYPE) BOTTOM_BYTE << shift));
    nodeValue |= ((GENOME_ITEM_TYPE) argumentValue << shift);
}

bool connectorsDiscartFirst(GENOME_ITEM_TYPE& connectorMask, int& connection) {
    int connectionIndex = 0;
    while ( connectionIndex < BITS_IN_ULONG && (( (GENOME_ITEM_TYPE) 1 << connectionIndex ) & connectorMask) == 0 ) {
        connectionIndex++;
    }
    if (connectionIndex >= BITS_IN_ULONG) {
        return false; // no existing connection
    } else {
        connectorMask ^= ((GENOME_ITEM_TYPE) 1 << connectionIndex); // discart connection
        connection = connectionIndex; // output connection
        return true; // connection found
    }
}

GENOME_ITEM_TYPE relativeToAbsoluteConnectorMask(GENOME_ITEM_TYPE relativeMask, int slot, int sizePreviousLayer, int connectorWidth) {
    GENOME_ITEM_TYPE absoluteMask = 0;
    GENOME_ITEM_TYPE baseRelativeOffset = (slot - (connectorWidth / 2) + sizePreviousLayer) % sizePreviousLayer;
    for (int bit = 0; bit < sizePreviousLayer; bit++) {
        if (relativeMask & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
            absoluteMask += pGlobals->precompPow[(baseRelativeOffset + bit + sizePreviousLayer) % sizePreviousLayer];
        }
    }
    return absoluteMask;
}

GENOME_ITEM_TYPE absoluteToRelativeConnectorMask(GENOME_ITEM_TYPE absoluteMask, int slot, int sizePreviousLayer, int connectorWidth) {
    GENOME_ITEM_TYPE relativeMask = 0;
    GENOME_ITEM_TYPE baseRelativeOffset = (slot - (connectorWidth / 2) + sizePreviousLayer) % sizePreviousLayer;
    for (int bit = 0; bit < sizePreviousLayer; bit++) {
        if (absoluteMask & (GENOME_ITEM_TYPE) pGlobals->precompPow[(bit + baseRelativeOffset + sizePreviousLayer) % sizePreviousLayer]) {
            relativeMask += pGlobals->precompPow[bit];
        }
    }
    return relativeMask;
}

unsigned char getNeutralValue(unsigned char function) {
    switch (function) {
    case FNC_NOP:
    case FNC_CONS:
    case FNC_OR:
    case FNC_XOR:
    case FNC_NOR:
    case FNC_NOT:
    case FNC_SHIL:
    case FNC_SHIR:
    case FNC_ROTL:
    case FNC_ROTR:
    case FNC_EQ:
    case FNC_LT:
    case FNC_GT:
    case FNC_LEQ:
    case FNC_GEQ:
    case FNC_BSLC:
    case FNC_READ:
    case FNC_EXT:
        return 0;
    case FNC_AND:
    case FNC_NAND:
        return UCHAR_MAX;
    }
    // unknown function constant
    mainLogger.out(LOGGER_ERROR) << "Neutral value undefined - unknown function (" << function << ")." << endl;
    return 0;
}

string functionToString(unsigned char function) {
    switch (function) {
    case FNC_NOP:   return "NOP";
    case FNC_CONS:  return "CONS";
    case FNC_AND:   return "AND";
    case FNC_NAND:  return "NAND";
    case FNC_OR:    return "OR";
    case FNC_XOR:   return "XOR";
    case FNC_NOR:   return "NOR";
    case FNC_NOT:   return "NOT";
    case FNC_SHIL:  return "ROTL";
    case FNC_SHIR:  return "ROTR";
    case FNC_ROTL:  return "CYCL";
    case FNC_ROTR:  return "CYCR";
    case FNC_EQ:    return "EQ";
    case FNC_LT:    return "LT";
    case FNC_GT:    return "GT";
    case FNC_LEQ:   return "LEQ";
    case FNC_GEQ:   return "GEQ";
    case FNC_BSLC:  return "BSLC";
    case FNC_READ:  return "READ";
    case FNC_EXT:   return "EXT";
    }
    // unknown function constant
    return string("UNKNOWN_") + to_string(function);
}
