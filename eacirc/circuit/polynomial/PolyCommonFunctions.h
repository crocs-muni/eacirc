#ifndef POLY_COMMON_FUNCTIONS_H
#define POLY_COMMON_FUNCTIONS_H

// One single element is unsigned long
// Originally term was here, but length of the term is set in configuration
// file, thus it is controlled on the higher level, not at the genome level.
#define POLY_GENOME_ITEM_TYPE GENOME_ITEM_TYPE

// Global definitions.
#include "EACglobals.h"

// We use 2D array genome (polynomials, terms).
#include <GAGenome.h>
#include <GA2DArrayGenome.h>

// Determine whether we are building for a 64-bit platform.
// _LP64: http://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html
#if defined(_M_X64) || defined(__amd64__) || defined(_LP64) || defined(_ILP64)
#define COMPILER_X64
#endif

/**
 * Function to evaluate one term element of an arbitrary size.
 * @param trm
 * @param input
 * @return
 */
unsigned int term_item_eval(POLY_GENOME_ITEM_TYPE trm, unsigned char * input);

// Fast ceiling function for integers.
#define OWN_CEIL(x)  (    (((int)(x)) < (x)) ? ((int)(x))+1 : ((int)(x))    )
#define OWN_FLOOR(x) (    (((int)(x)) < (x)) ? ((int)(x))-1 : ((int)(x))    )

#endif // POLY_COMMON_FUNCTIONS_H
