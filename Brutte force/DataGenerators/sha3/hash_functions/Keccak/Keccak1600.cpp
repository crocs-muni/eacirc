#include "Keccak1600.h"
#include <iostream>
extern "C" {
#include "sha3/hash_functions/Keccak64_common/KeccakF-1600-interface.h"
#include "sha3/hash_functions/Keccak64_common/KeccakSponge.h"
}
using namespace std;

Keccak1600::Keccak1600() : rounds(1) {

}

Keccak1600::~Keccak1600() {

}

int Keccak1600::setNumRounds(int rounds) { 
    if (rounds!=24 && (rounds<1 || rounds>8)){
        return -1;
    }
    
    this->rounds=rounds; 
    return 1; 
}

int Keccak1600::evaluate(const unsigned char* input, unsigned char* output) const {
    spongeState state;
    InitSponge(&state, 1024, 576);
    Absorb(&state, input, 1022, rounds);
    Squeeze(&state, output, 1024, rounds);
    return 1;
}

int Keccak1600::evaluate(const unsigned char* input, const unsigned char* key, unsigned char* output) const {
    spongeState state;
    InitSponge(&state, 1024, 576);
    Absorb(&state, input, 1022, rounds);
    Squeeze(&state, output, 1024, rounds);
    return 1;
}
