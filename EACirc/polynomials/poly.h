#ifndef _EACIRC_POLY_H
#define _EACIRC_POLY_H

// One single element is unsigned long
// Originally term was here, but length of the term is set in configuration
// file, thus it is controlled on the higher level, not at the genome level.
#define POLY_GENOME_ITEM_TYPE unsigned long

// Global definitions.
#include "EACglobals.h"

// We use 2D array genome (polynomials, terms).
#include "../galib/GA2DArrayGenome.h"

/**
 * Function to evaluate one term element of an arbitrary size.
 * @param trm
 * @param input
 * @return 
 */
unsigned int term_item_eval(POLY_GENOME_ITEM_TYPE trm, unsigned char * input);

#define TERM_ITEM_EVAL_1(trm, input)                                             \
    ((((*input)     & ((trm)     & 0xfful)) == ((trm)     & 0xfful)))

#define TERM_ITEM_EVAL_2(trm, input)                                           \
    (((*((input)+0) & ((trm)     & 0xfful)) == ((trm)     & 0xfful)) &         \
    ( (*((input)+1) & ((trm>>8)  & 0xfful)) == ((trm>>8)  & 0xfful)))

#define TERM_ITEM_EVAL_4(trm, input)                                           \
    (((*((input)+0) & ((trm)     & 0xfful)) == ((trm)     & 0xfful)) &         \
    ( (*((input)+1) & ((trm>>8)  & 0xfful)) == ((trm>>8)  & 0xfful)) &         \
    ( (*((input)+2) & ((trm>>16) & 0xfful)) == ((trm>>16) & 0xfful)) &         \
    ( (*((input)+3) & ((trm>>24) & 0xfful)) == ((trm>>24) & 0xfful)))

#define TERM_ITEM_EVAL_8(trm, input)                                           \
    (((*((input)+0) & ((trm)     & 0xfful)) == ((trm)     & 0xfful)) &         \
    ( (*((input)+1) & ((trm>>8)  & 0xfful)) == ((trm>>8)  & 0xfful)) &         \
    ( (*((input)+2) & ((trm>>16) & 0xfful)) == ((trm>>16) & 0xfful)) &         \
    ( (*((input)+3) & ((trm>>24) & 0xfful)) == ((trm>>24) & 0xfful)) &         \
    ( (*((input)+4) & ((trm>>32) & 0xfful)) == ((trm>>32) & 0xfful)) &         \
    ( (*((input)+5) & ((trm>>40) & 0xfful)) == ((trm>>40) & 0xfful)) &         \
    ( (*((input)+6) & ((trm>>48) & 0xfful)) == ((trm>>48) & 0xfful)) &         \
    ( (*((input)+7) & ((trm>>56) & 0xfful)) == ((trm>>56) & 0xfful)))

#define TERM_ITEM_EVAL_16(trm, input) term_item_eval(trm, input)
#define TERM_ITEM_EVAL_32(trm, input) term_item_eval(trm, input)
#define TERM_ITEM_EVAL_64(trm, input) term_item_eval(trm, input)
#define TERM_ITEM_EVAL_N(trm, input) term_item_eval(trm, input)

//#define TERM_ITEM_EVAL(size, trm, input) TERM_ITEM_EVAL_##size(trm, input)

#endif // end of file
