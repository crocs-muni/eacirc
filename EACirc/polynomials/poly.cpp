#include "poly.h"

unsigned int term_item_eval(POLY_GENOME_ITEM_TYPE trm, unsigned char * input){
    unsigned int res=1;
    for(unsigned int i=0; i<sizeof(POLY_GENOME_ITEM_TYPE); i++){
        const unsigned char mask = (trm>>(8*i)) & 0xfful;
        res &= (*((input)+i) & mask) == mask;
    }
    
    return res;
}






