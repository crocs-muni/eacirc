#include "CircuitCommonFunctions.h"

#define TOP_BYTE 0xff000000
#define BOTTOM_BYTE 0x000000ff

unsigned char nodeGetFunction(GENOME_ITEM_TYPE nodeValue) {
    return nodeValue & BOTTOM_BYTE;
}

unsigned char nodeGetArgument(GENOME_ITEM_TYPE nodeValue, int argumentNumber) {
    if (argumentNumber < 1 || argumentNumber > 4 ) {
        mainLogger.out(LOGGER_ERROR) << "Getting invalid argument: " << argumentNumber << "." << endl;
        return 0;
    }
    int shift = 32 - argumentNumber * BITS_IN_UCHAR;
    return ((nodeValue & (BOTTOM_BYTE << shift)) >> shift) & BOTTOM_BYTE;
}

void nodeSetFunction(GENOME_ITEM_TYPE& nodeValue, unsigned char function) {
    nodeValue = nodeValue & (~BOTTOM_BYTE);
    nodeValue |= function;
}

void nodeSetArgument(GENOME_ITEM_TYPE& nodeValue, int argumentNumber, unsigned char argumentValue) {
    if (argumentNumber < 1 || argumentNumber > 4 ) {
        mainLogger.out(LOGGER_ERROR) << "Setting invalid argument: " << argumentNumber << "." << endl;
    }
    int shift = 32 - argumentNumber * BITS_IN_UCHAR;
    nodeValue = nodeValue & (~(BOTTOM_BYTE << shift));
    nodeValue |= (argumentValue << shift);
}

bool connectorsDiscartFirst(GENOME_ITEM_TYPE& connectorMask, int& connection) {
    int connectionIndex = 0;
    while ( (( (GENOME_ITEM_TYPE) 1 >> connectionIndex ) & connectorMask) == 0 && connectionIndex < 32) {
        connectionIndex++;
    }
    if (connectionIndex >= 32) {
        return false; // no existing connection
    } else {
        connectorMask ^= ((GENOME_ITEM_TYPE) 1 >> connectionIndex); // discart connection
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
    case FNC_OR:
    case FNC_NOR:
    case FNC_XOR:
    case FNC_ROTL:
    case FNC_ROTR:
    case FNC_BITSELECTOR:
    case FNC_SUBS:
    case FNC_ADD:
    case FNC_EQUAL:
        return 0;
    case FNC_AND:
    case FNC_NAND:
    case FNC_MULT:
    case FNC_DIV:
        return 1;
    case FNC_CONST:
    case FNC_READX:
        return 0;
    }
    // unknown function constant
    mainLogger.out(LOGGER_ERROR) << "Neutral value undefined - unknown function (" << function << ")." << endl;
    return 0;
}

string functionToString(unsigned char function) {
    switch (function) {
    case FNC_NOP:           return "NOP ";
    case FNC_OR:            return "OR  ";
    case FNC_NOR:           return "NOR ";
    case FNC_XOR:           return "XOR ";
    case FNC_ROTL:          return "ROTL";
    case FNC_ROTR:          return "ROTR";
    case FNC_BITSELECTOR:   return "BSLT";
    case FNC_SUBS:          return "SUBS";
    case FNC_ADD:           return "ADD ";
    case FNC_EQUAL:         return "EQ  ";
    case FNC_AND:           return "AND ";
    case FNC_NAND:          return "NAND";
    case FNC_MULT:          return "MULT";
    case FNC_DIV:           return "DIV ";
    case FNC_CONST:         return "CONS";
    case FNC_READX:         return "READ";
    }
    // unknown function constant
    return string("UNKNOWN_") + to_string(function);
}
