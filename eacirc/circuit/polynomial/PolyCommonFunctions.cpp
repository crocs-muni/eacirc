#include "PolyCommonFunctions.h"

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
unsigned int term_item_eval(POLY_GENOME_ITEM_TYPE trm, unsigned char * input){
    unsigned int res=1;
    for(unsigned int i=0; i<sizeof(POLY_GENOME_ITEM_TYPE); i++){
        const unsigned char mask = (trm>>(8*i)) & 0xfful;
        // If mask is null, do not process this input.
        // It may happen the 8*POLY_GENOME_ITEM_TYPE is bigger than
        // number of variables, thus we would read ahead of input array.
        // Term itself must not contain variables out of the range (guarantees
        // that an invalid memory is not read).
        if (mask == 0) continue;

        res &= (*((input)+i) & mask) == mask;
    }

    return res;
}






