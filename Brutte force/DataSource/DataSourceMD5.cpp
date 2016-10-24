//
// Created by Dusan Klinec on 19.06.16.
//

#include "DataSourceMD5.h"
#include <cstring>
#include <random>

DataSourceMD5::DataSourceMD5(unsigned long seed, int Nr) {
    std::minstd_rand systemGenerator((unsigned int) seed);
    for (unsigned char i = 0; i < MD5_DIGEST_LENGTH; i++) {
        m_md5Accumulator[i] = (unsigned char)systemGenerator();
    }
    m_rounds = Nr;
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
void DataSourceMD5::read(char *buffer, char* messages, size_t size){
    int messageoffset = 0;
    for(size_t offset = 0; offset < size; offset += MD5_DIGEST_LENGTH){

        memcpy(m_md5Accumulator, messages + messageoffset, MD5_DIGEST_LENGTH);
        messageoffset += MD5_BLOCK_LENGTH;
        updateAccumulator();
        memcpy(buffer+offset, m_md5Accumulator, std::min((size_t)MD5_DIGEST_LENGTH, size-offset));
    }

}


std::string DataSourceMD5::desc() {
    return "MD5";
}

int DataSourceMD5::updateAccumulator() {
    MD5_CTX mdContext;
    MD5Init(&mdContext, m_rounds);
    MD5Update(&mdContext, m_md5Accumulator, MD5_DIGEST_LENGTH, m_rounds);
    MD5Final(&mdContext,m_rounds);
    memcpy(m_md5Accumulator, mdContext.digest, MD5_DIGEST_LENGTH);
    return 0;
}
