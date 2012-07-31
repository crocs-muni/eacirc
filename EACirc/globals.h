#ifndef GLOBALS_H
#define GLOBALS_H

#ifndef HIGHBYTE
    #define HIGHBYTE(x)  x >> 8 
#endif

#ifndef LOWBYTE
    #define LOWBYTE(x)   x & 0xFF 
#endif

#define FILETIME_TO_SECOND                      10000000 

#define MAX_INI_VALUE_CHAR                      512

#define FLAG_OS_WIN9X                           1
#define FLAG_OS_WINNT                           2
    
//#define _NO_AVAIL_CARD


typedef struct CARDAPDU {
	unsigned char   cla;
	unsigned char   ins;
	unsigned char   p1;
	unsigned char   p2;
	unsigned char   lc;
	unsigned char   le;
	unsigned char   DataIn[256];
	unsigned char   DataOut[256];
	unsigned short  sw;
} CARDAPDU;

#define SLOT_ANY_AVAILABLE	-1

#endif