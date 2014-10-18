#include "poly.h"

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






