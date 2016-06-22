//
// Created by Dusan Klinec on 21.06.16.
//

#include "DataSourceRC4Column.h"
#include <random>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include "../DataGenerators/arcfour.h"

DataSourceRC4Column::DataSourceRC4Column(unsigned long seed, unsigned blockSize, unsigned keySize) {
    if (blockSize <= 0){
        throw std::out_of_range("Invalid block value");
    }

    if(keySize > RC4_STATE_SIZE){
        throw std::out_of_range("Maximum key size is 256B");
    }

    this->m_blockSize = blockSize;
    this->m_keySize = 16;
    this->m_gen = new std::minstd_rand((unsigned int) seed);
}

DataSourceRC4Column::~DataSourceRC4Column() {
    if (this->m_gen != nullptr) {
        delete this->m_gen;
        this->m_gen = nullptr;
    }
}

long long DataSourceRC4Column::getAvailableData() {
    return 9223372036854775807; // 2^63-1
}

void DataSourceRC4Column::read(char *buffer, size_t size) {
    BYTE rc4State[RC4_STATE_SIZE];
    BYTE rc4Key[RC4_STATE_SIZE];

    for(size_t offset = 0; offset < size; offset += this->m_blockSize){
        // Each time generate new key and initialize a new state.
        for (unsigned i = 0; i < this->m_keySize; i++) {
            rc4Key[i] = (uint8_t) this->m_gen->operator()();
        }
        arcfour_key_setup(rc4State, rc4Key, this->m_keySize);

        // Generate key stream.
        arcfour_generate_stream(rc4State, (BYTE*)(buffer+offset), std::min((size_t)this->m_blockSize, size-offset));
    }
}

std::string DataSourceRC4Column::desc() {
    std::stringstream ss;
    ss << "RC4Column-key" << this->m_keySize << "-block" << this->m_blockSize;
    return ss.str();
}
