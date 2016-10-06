//
// Created by Dusan Klinec on 19.06.16.
//

#include "DataSourceAES.h"
#include <random>
#include <cstring>
#include <sstream>
#include "../DataGenerators/aes.h"


DataSourceAES::DataSourceAES(unsigned long seed, int rounds) {
    std::minstd_rand systemGenerator((unsigned int) seed);
    for (unsigned char i = 0; i < 16; i++) {
        m_key[i] = (uint8_t) systemGenerator();
        m_iv[i] = (uint8_t) systemGenerator();
    }
    m_rounds = rounds;
    m_counter = systemGenerator();
}

long long DataSourceAES::getAvailableData() {
    return 9223372036854775807; // 2^63-1
}

void DataSourceAES::read(char *buffer, size_t size) {
    const int workingBufferSize = 16;
    uint8_t workingBufferIn[workingBufferSize];
    memset(workingBufferIn, 0, workingBufferSize);

    //counter mode
    for(size_t offset = 0; offset < size; offset += workingBufferSize){
        memcpy(workingBufferIn, &m_counter, sizeof(m_counter));
        // Encrypt input buffer.
        AES128_ECB_encrypt(workingBufferIn, m_key, (uint8_t*)(buffer + offset), m_rounds);
        ++m_counter;
    }
}

std::string DataSourceAES::desc() {
    std::stringstream ss;
    ss << "AES-CTR-r" << m_rounds;
    return ss.str();
}

