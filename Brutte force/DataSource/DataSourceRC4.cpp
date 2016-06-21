//
// Created by Dusan Klinec on 21.06.16.
//

#include "DataSourceRC4.h"
#include <random>
#include <cstring>
#include "../DataGenerators/arcfour.h"

DataSourceRC4::DataSourceRC4(unsigned long seed, unsigned keySize) {
    std::minstd_rand systemGenerator((unsigned int) seed);

    if(keySize > RC4_STATE_SIZE){
        throw std::out_of_range("Maximum key size is 256B");
    }

    this->keySize = keySize;
    for (unsigned i = 0; i < this->keySize; i++) {
        m_key[i] = (uint8_t) systemGenerator();
    }

    arcfour_key_setup(m_state, m_key, this->keySize);
}

long long DataSourceRC4::getAvailableData() {
    return 9223372036854775807; // 2^63-1
}

void DataSourceRC4::read(char *buffer, size_t size) {
    const int workingBufferSize = 256;
    uint8_t workingBufferOu[workingBufferSize];

    for(size_t offset = 0; offset < size; offset += workingBufferSize){
        // Generate key stream.
        arcfour_generate_stream(m_state, workingBufferOu, workingBufferSize);
        // Copy workingBufferSizeB to the output buffer.
        memcpy(buffer+offset, workingBufferOu, std::min((size_t)workingBufferSize, size-offset));
    }
}

std::string DataSourceRC4::desc() {
    return "RC4";
}

