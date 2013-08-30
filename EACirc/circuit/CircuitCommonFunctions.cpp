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

GENOME_ITEM_TYPE relativeToAbsoluteConnectorMask(GENOME_ITEM_TYPE relativeMask, int slot, int sizePreviousLayer, int connectorWidth) {
    GENOME_ITEM_TYPE absoluteMask = 0;
    GENOME_ITEM_TYPE baseRelativeOffset = (slot - (connectorWidth / 2) + sizePreviousLayer) % sizePreviousLayer;
    for (int bit = 0; bit < sizePreviousLayer ; bit++) {
        if (relativeMask & (GENOME_ITEM_TYPE) pGlobals->precompPow[bit]) {
            absoluteMask += pGlobals->precompPow[(baseRelativeOffset + bit + sizePreviousLayer) % sizePreviousLayer];
        }
    }
    return absoluteMask;
}

GENOME_ITEM_TYPE absoluteToRelativeConnectorMask(GENOME_ITEM_TYPE absoluteMask, int slot, int sizePreviousLayer, int connectorWidth) {

}
