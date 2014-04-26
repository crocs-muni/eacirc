#include "poly.h"

int term_item_eval(POLY_GENOME_ITEM_TYPE trm, unsigned char * input){
    int res=1;
    for(int i=0; i<sizeof(POLY_GENOME_ITEM_TYPE); i++){
        const unsigned char mask = (trm>>(8*i)) & 0xfful;
        res &= (*((input)+i) & mask) == mask;
    }
    
    return res;
}






