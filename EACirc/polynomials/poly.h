#ifndef _EACIRC_POLY_H
#define _EACIRC_POLY_H

// One single element is unsigned long
// Originally term was here, but length of the term is set in configuration
// file, thus it is controlled on the higher level, not at the genome level.
#define POLY_GENOME_ITEM_TYPE unsigned long

// Global definitions.
#include "EACglobals.h"

// We use 2D array genome (polynomials, terms).
#include "GA2DArrayGenome.h"

/**
 * Function to evaluate one term element of an arbitrary size.
 * @param trm
 * @param input
 * @return 
 */
int term_item_eval(POLY_GENOME_ITEM_TYPE trm, unsigned char * input);

#if (sizeof(POLY_GENOME_ITEM_TYPE)==1)
#define TERM_ITEM_EVAL(trm, input)                                             \
    ((((*input)     & ((trm)     & 0xfful)) == ((trm)     & 0xfful)))
#elif (sizeof(POLY_GENOME_ITEM_TYPE)==2)
#define TERM_ITEM_EVAL(trm, input)                                             \
    (((*((input)+0) & ((trm)     & 0xfful)) == ((trm)     & 0xfful)) &         \
    ( (*((input)+1) & ((trm>>8)  & 0xfful)) == ((trm>>8)  & 0xfful)))
#elif (sizeof(POLY_GENOME_ITEM_TYPE)==4)
#define TERM_ITEM_EVAL(trm, input)                                             \
    (((*((input)+0) & ((trm)     & 0xfful)) == ((trm)     & 0xfful)) &         \
    ( (*((input)+1) & ((trm>>8)  & 0xfful)) == ((trm>>8)  & 0xfful)) &         \
    ( (*((input)+2) & ((trm>>16) & 0xfful)) == ((trm>>16) & 0xfful)) &         \
    ( (*((input)+3) & ((trm>>24) & 0xfful)) == ((trm>>24) & 0xfful)))
#elif (sizeof(POLY_GENOME_ITEM_TYPE)==8)
#define TERM_ITEM_EVAL(trm, input)                                             \
    (((*((input)+0) & ((trm)     & 0xfful)) == ((trm)     & 0xfful)) &         \
    ( (*((input)+1) & ((trm>>8)  & 0xfful)) == ((trm>>8)  & 0xfful)) &         \
    ( (*((input)+2) & ((trm>>16) & 0xfful)) == ((trm>>16) & 0xfful)) &         \
    ( (*((input)+3) & ((trm>>24) & 0xfful)) == ((trm>>24) & 0xfful)) &         \
    ( (*((input)+4) & ((trm>>32) & 0xfful)) == ((trm>>32) & 0xfful)) &         \
    ( (*((input)+5) & ((trm>>40) & 0xfful)) == ((trm>>40) & 0xfful)) &         \
    ( (*((input)+6) & ((trm>>48) & 0xfful)) == ((trm>>48) & 0xfful)) &         \
    ( (*((input)+7) & ((trm>>56) & 0xfful)) == ((trm>>56) & 0xfful)))
#else
#warning "Term internal type has non-standard size!"
#define TERM_ITEM_EVAL(trm, input) term_item_eval(trm, input)
#endif


#endif // end of file
