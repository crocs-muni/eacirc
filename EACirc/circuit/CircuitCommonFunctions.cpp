#include "CircuitCommonFunctions.h"

#define TOP_BYTE 0xff000000
#define BOTTOM_BYTE 0x000000ff

unsigned char nodeGetFunction(GENOME_ITEM_TYPE nodeValue) {
    return nodeValue & BOTTOM_BYTE;
}

unsigned char nodeGetArgument1(GENOME_ITEM_TYPE nodeValue) {
    return ((nodeValue & TOP_BYTE) >> 24) & BOTTOM_BYTE;
}

void nodeSetFunction(GENOME_ITEM_TYPE* nodeValue, unsigned char function) {
    *nodeValue = *nodeValue & (~BOTTOM_BYTE);
    *nodeValue |= function;
}

void nodeSetArgument1(GENOME_ITEM_TYPE* nodeValue, unsigned char argument1) {
    *nodeValue = *nodeValue & (~TOP_BYTE);
    *nodeValue |= (argument1 << 24);
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
    //case FNC_SUM:
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
    case FNC_NOP:           return "FNC_NOP";
    case FNC_OR:            return "FNC_OR";
    case FNC_NOR:           return "FNC_NOR";
    case FNC_XOR:           return "FNC_XOR";
    case FNC_ROTL:          return "FNC_ROTL";
    case FNC_ROTR:          return "FNC_ROTR";
    case FNC_BITSELECTOR:   return "FNC_BITSELECTOR";
    //case FNC_SUM:           return "FNC_SUM";
    case FNC_SUBS:          return "FNC_SUBS";
    case FNC_ADD:           return "FNC_ADD";
    case FNC_EQUAL:         return "FNC_EQUAL";
    case FNC_AND:           return "FNC_AND";
    case FNC_NAND:          return "FNC_NAND";
    case FNC_MULT:          return "FNC_MULT";
    case FNC_DIV:           return "FNC_DIV";
    case FNC_CONST:         return "FNC_CONST";
    case FNC_READX:         return "FNC_READX";
    }
    // unknown function constant
    return string("UNKNOWN_") + to_string(function);
}
