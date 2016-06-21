//
// Created by Dusan Klinec on 21.06.16.
//

#include "DataSourceRC4Column.h"
#include <random>
#include <cstring>
#include "../DataGenerators/arcfour.h"

DataSourceRC4Column::DataSourceRC4Column(unsigned long seed, unsigned blockSize, unsigned keySize) {
    if (blockSize > RC4_STATE_SIZE){
        throw std::out_of_range("Block size maximum is 256B for this mode");
    } else if (blockSize <= 0){
        throw std::out_of_range("Invalid block value");
    }

    if(keySize > RC4_STATE_SIZE){
        throw std::out_of_range("Maximum key size is 256B");
    }

    this->blockSize = blockSize;
    this->keySize = 16;
    this->gen = new std::minstd_rand((unsigned int) seed);
}

DataSourceRC4Column::~DataSourceRC4Column() {
    if (this->gen != nullptr) {
        delete this->gen;
        this->gen = nullptr;
    }
}

long long DataSourceRC4Column::getAvailableData() {
    return 9223372036854775807; // 2^63-1
}

void DataSourceRC4Column::read(char *buffer, size_t size) {
    const int workingBufferSize = RC4_STATE_SIZE;
    BYTE rc4State[RC4_STATE_SIZE];
    BYTE rc4Key[RC4_STATE_SIZE];
    uint8_t workingBufferOu[workingBufferSize];

    for(size_t offset = 0; offset < size; offset += this->blockSize){
        // Each time generate new key and initialize a new state.
        for (unsigned i = 0; i < this->keySize; i++) {
            rc4Key[i] = (uint8_t) this->gen->operator()();
        }
        arcfour_key_setup(rc4State, rc4Key, this->keySize);

        // Generate key stream.
        arcfour_generate_stream(rc4State, workingBufferOu, (size_t) this->blockSize);
        // Copy workingBufferSizeB to the output buffer.
        memcpy(buffer+offset, workingBufferOu, std::min((size_t)this->blockSize, size-offset));
    }
}

std::string DataSourceRC4Column::desc() {
    return "RC4Column";
}
