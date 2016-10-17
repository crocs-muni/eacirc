//
// Created by Dusan Klinec on 21.06.16.
//

#include "DataSourceRC4.h"
#include <random>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include "../DataGenerators/arcfour.h"

DataSourceRC4::DataSourceRC4(unsigned long seed, unsigned keySize) {
    std::minstd_rand systemGenerator((unsigned int) seed);

    if(keySize > RC4_STATE_SIZE){
        throw std::out_of_range("Maximum key size is 256B");
    }

    this->m_keySize = keySize;
    for (unsigned i = 0; i < this->m_keySize; i++) {
        m_key[i] = (uint8_t) systemGenerator();
    }

    arcfour_key_setup(m_state, m_key, this->m_keySize);
}

long long DataSourceRC4::getAvailableData() {
    return 9223372036854775807; // 2^63-1
}

void DataSourceRC4::read(char *buffer, char *key, size_t size, size_t BytesPerIter) {
    for(size_t offset = 0; offset < size; offset += this->m_keySize){
        //load key from array
        memcpy(m_key, key + offset, m_keySize );
        //keysetup
        arcfour_key_setup(m_state, m_key, this->m_keySize);
        // bytesPerIter of keystream to buffer.
        arcfour_generate_stream(m_state, (BYTE*)buffer, BytesPerIter);
    }
}

void DataSourceRC4::read(char *buffer, size_t size) {
    arcfour_generate_stream(m_state, (BYTE*)buffer, size);
}

std::string DataSourceRC4::desc() {
    std::stringstream ss;
    ss << "RC4-key" << this->m_keySize;
    return ss.str();
}

