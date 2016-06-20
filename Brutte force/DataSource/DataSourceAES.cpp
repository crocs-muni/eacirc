//
// Created by Dusan Klinec on 19.06.16.
//

#include "DataSourceAES.h"
#include <random>
#include "../DataGenerators/aes.h"

DataSourceAES::DataSourceAES(unsigned long seed) {
    std::minstd_rand systemGenerator((unsigned int) seed);
    for (unsigned char i = 0; i < 16; i++) {
        m_key[i] = (uint8_t) systemGenerator();
        m_iv[i] = (uint8_t) systemGenerator();
    }
}

long long DataSourceAES::getAvailableData() {
    return 9223372036854775807; // 2^63-1
}

void DataSourceAES::read(char *buffer, size_t size) {
    const int workingBufferSize = 256;
    uint8_t workingBufferIn[workingBufferSize+16];
    uint8_t workingBufferOu[workingBufferSize+16];
    for(int i=0; i<workingBufferSize+16; i++){
        workingBufferIn[i] = 0;
    }

    for(size_t offset = 0; offset < size; offset += workingBufferSize){
        AES128_CBC_encrypt_buffer(workingBufferOu, workingBufferIn, workingBufferSize+16, m_key, m_iv);
        memcpy(m_iv, workingBufferOu + workingBufferSize, 16);

        memcpy(buffer+offset, workingBufferOu, std::max((size_t)workingBufferSize, size-offset));
    }
}

std::string DataSourceAES::desc() {
    return "AES";
}

