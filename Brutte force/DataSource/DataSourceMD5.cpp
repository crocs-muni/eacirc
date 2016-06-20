//
// Created by Dusan Klinec on 19.06.16.
//

#include "DataSourceMD5.h"
#include <cstring>
#include <random>

DataSourceMD5::DataSourceMD5(unsigned long seed) {
    std::minstd_rand systemGenerator((unsigned int) seed);
    for (unsigned char i = 0; i < MD5_DIGEST_LENGTH; i++) {
        m_md5Accumulator[i] = (unsigned char)systemGenerator();
    }
}

long long DataSourceMD5::getAvailableData() {
    return 9223372036854775807; // 2^63-1
}

void DataSourceMD5::read(char *buffer, size_t size) {
    for(size_t offset = 0; offset < size; offset += MD5_DIGEST_LENGTH){
        updateAccumulator();
        memcpy(buffer+offset, m_md5Accumulator, std::min((size_t)MD5_DIGEST_LENGTH, size-offset));
    }
}

std::string DataSourceMD5::desc() {
    return "MD5";
}

int DataSourceMD5::updateAccumulator() {
    MD5_CTX mdContext;
    MD5Init(&mdContext);
    MD5Update(&mdContext, m_md5Accumulator, MD5_DIGEST_LENGTH);
    MD5Final(&mdContext);
    memcpy(m_md5Accumulator, mdContext.digest, MD5_DIGEST_LENGTH);
    return 0;
}
