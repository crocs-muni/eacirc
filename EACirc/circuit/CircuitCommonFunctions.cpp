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
