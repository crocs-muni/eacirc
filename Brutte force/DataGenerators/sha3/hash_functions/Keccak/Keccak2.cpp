// keccak.c
// 19-Nov-11  Markku-Juhani O. Saarinen <mjos@iki.fi>
// A baseline Keccak (3rd round) implementation.

#include "Keccak2.h"
#include <iostream>
using namespace std;

const uint64_t keccakf_rndc[24] = 
{
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

const int keccakf_rotc[24] = 
{
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14, 
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

const int keccakf_piln[24] = 
{
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4, 
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1 
};

Keccak2::Keccak2() : rounds(24), r(1088), c(512) {

}

Keccak2::~Keccak2() {

}

// update the state with given number of rounds
void Keccak2::keccakf(uint64_t st[25], int rounds) const
{
    int i, j, round;
    uint64_t t, bc[5];

    for (round = 0; round < rounds; round++) {

        // Theta
        for (i = 0; i < 5; i++)     
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        // Rho Pi
        t = st[1];
        for (i = 0; i < 24; i++) {
            j = keccakf_piln[i];
            bc[0] = st[j];
            st[j] = ROTL64(t, keccakf_rotc[i]);
            t = bc[0];
        }

        //  Chi
        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++)
                bc[i] = st[j + i];
            for (i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        //  Iota
        st[0] ^= keccakf_rndc[round];
    }
}

// compute a keccak hash (md) of given byte length from "in"
int Keccak2::keccak(const uint8_t *in, int inlen, uint8_t *md, int mdlen) const
{
    int i, rsiz, rsizw;
    rsiz = r;
    rsizw = rsiz / 8;
    
    uint64_t st[rsizw];    
    uint8_t temp[rsiz];
    
    memset(st, 0, sizeof(st));

    for ( ; inlen >= rsiz; inlen -= rsiz, in += rsiz) {
        for (i = 0; i < rsizw; i++){
            st[i] ^= ((uint64_t *) in)[i];
        }
        keccakf(st, rounds);
    }
    
    // last block and padding
    memcpy(temp, in, inlen);
    temp[inlen++] = 1;
    memset(temp + inlen, 0, rsiz - inlen);
    temp[rsiz - 1] |= 0x80;

    for (i = 0; i < rsizw; i++){
        st[i] ^= ((uint64_t *) temp)[i];
    }
        
    keccakf(st, rounds);

    memcpy(md, st, mdlen);

    return 0;
}

int Keccak2::evaluate(const unsigned char *input, const unsigned char *key, unsigned char *output ) const {
    unsigned char in[200] = {0};

    memcpy(in,      input, 16);
    memcpy(in + 16, key,   16);

    keccak(in, 200, output, 16);

    return 1;
}

int Keccak2::evaluate(const unsigned char* input, unsigned char* output) const {
    unsigned char in[200] = {0};

    memcpy(in,input,32);
    keccak(in,200,output,16);
    
    return 1;
}

void Keccak2::set_parameters(int r, int c) {
    this->r = r;
    this->c = c;
}