#ifndef _EACIRC_POLY_H
#define _EACIRC_POLY_H

// One single element is unsigned long
// Originally term was here, but length of the term is set in configuration
// file, thus it is controlled on the higher level, not at the genome level.
#define POLY_GENOME_ITEM_TYPE GENOME_ITEM_TYPE

// Global definitions.
#include "EACglobals.h"

// We use 2D array genome (polynomials, terms).
#include "../galib/GAGenome.h"
#include "../galib/GA2DArrayGenome.h"

/**
 * Function to evaluate one term element of an arbitrary size.
 * @param trm
 * @param input
 * @return 
 */
unsigned int term_item_eval(POLY_GENOME_ITEM_TYPE trm, unsigned char * input);

// Shifts term to the right 8*shift, and takes the lower 8 bits (corresponds to the input
// representation with unsigned char).
#define MASK_TERM(trm, shift) (((trm)>>(8*(shift))) & 0xfful)

#define TERM_ITEM_EVAL_1(trm, input)                                                           \
    (MASK_TERM(trm, 0) == 0 ? 1 : (((*(input)) & MASK_TERM(trm, 0)) == MASK_TERM(trm, 0)))

#define TERM_ITEM_EVAL_2(trm, input) (                                                         \
    (MASK_TERM(trm, 0) == 0 ? 1 : (((*((input)+0)) & MASK_TERM(trm, 0)) == MASK_TERM(trm, 0))) & \
    (MASK_TERM(trm, 1) == 0 ? 1 : (((*((input)+1)) & MASK_TERM(trm, 1)) == MASK_TERM(trm, 1))) )

#define TERM_ITEM_EVAL_4(trm, input) (                                                         \
    (MASK_TERM(trm, 0) == 0 ? 1 : (((*((input)+0)) & MASK_TERM(trm, 0)) == MASK_TERM(trm, 0))) & \
    (MASK_TERM(trm, 1) == 0 ? 1 : (((*((input)+1)) & MASK_TERM(trm, 1)) == MASK_TERM(trm, 1))) & \
    (MASK_TERM(trm, 2) == 0 ? 1 : (((*((input)+2)) & MASK_TERM(trm, 2)) == MASK_TERM(trm, 2))) & \
    (MASK_TERM(trm, 3) == 0 ? 1 : (((*((input)+3)) & MASK_TERM(trm, 3)) == MASK_TERM(trm, 3))) )

//
// Term evaluation operation actually used in the experiment for unsigned long type (sizeof(unsigned long)==8).
// Evaluation: 
//
//     7        6        5        4        3        2        1        0
// +----------------------------------------------------------------------+
// | in[7] |  in[6] |  in[5] |  in[4] |  in[3] |  in[2] |  in[1] |  in[0] | unsigned char * input
// +----------------------------------------------------------------------+ 
//     &        &        &        &        &        &        &        &   
// +----------------------------------------------------------------------+
// | tr[7] |  tr[6] |  tr[5] |  tr[4] |  tr[3] |  tr[2] |  tr[1] |  tr[0] | POLY_GENOME_ITEM_TYPE
// +----------------------------------------------------------------------+
// |                                                                     |
// x_63                                                                 x_0
//
// tr[i] == MASK_TERM(tr, i)
// 
#define TERM_ITEM_EVAL_8(trm, input) (                                                         \
    (MASK_TERM(trm, 0) == 0 ? 1 : (((*((input)+0)) & MASK_TERM(trm, 0)) == MASK_TERM(trm, 0))) & \
    (MASK_TERM(trm, 1) == 0 ? 1 : (((*((input)+1)) & MASK_TERM(trm, 1)) == MASK_TERM(trm, 1))) & \
    (MASK_TERM(trm, 2) == 0 ? 1 : (((*((input)+2)) & MASK_TERM(trm, 2)) == MASK_TERM(trm, 2))) & \
    (MASK_TERM(trm, 3) == 0 ? 1 : (((*((input)+3)) & MASK_TERM(trm, 3)) == MASK_TERM(trm, 3))) & \
    (MASK_TERM(trm, 4) == 0 ? 1 : (((*((input)+4)) & MASK_TERM(trm, 4)) == MASK_TERM(trm, 4))) & \
    (MASK_TERM(trm, 5) == 0 ? 1 : (((*((input)+5)) & MASK_TERM(trm, 5)) == MASK_TERM(trm, 5))) & \
    (MASK_TERM(trm, 6) == 0 ? 1 : (((*((input)+6)) & MASK_TERM(trm, 6)) == MASK_TERM(trm, 6))) & \
    (MASK_TERM(trm, 7) == 0 ? 1 : (((*((input)+7)) & MASK_TERM(trm, 7)) == MASK_TERM(trm, 7))) )

#define TERM_ITEM_EVAL_16(trm, input) term_item_eval(trm, input)
#define TERM_ITEM_EVAL_32(trm, input) term_item_eval(trm, input)
#define TERM_ITEM_EVAL_64(trm, input) term_item_eval(trm, input)
#define TERM_ITEM_EVAL_N(trm, input)  term_item_eval(trm, input)

// Integer has to be used in the third parameter.
#define TERM_ITEM_EVAL(trm, input, size) TERM_ITEM_EVAL_##size(trm, input)

//#define TERM_ITEM_EVAL(size, trm, input) TERM_ITEM_EVAL_##size(trm, input)

#endif // end of file
