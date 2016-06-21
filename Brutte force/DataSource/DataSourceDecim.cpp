//
// Created by Dusan Klinec on 21.06.16.
//

#include <sstream>
#include "DataSourceDecim.h"
#define DECIM_DEFAULT_IV_LEN 16
#define DECIM_DEFAULT_KEY_LEN 16

DataSourceDecim::DataSourceDecim(unsigned long seed, int rounds) {
    this->m_gen = new std::minstd_rand((unsigned int) seed);
    this->m_rounds = (unsigned)rounds;
    this->m_cipher = new ECRYPT_Decim();
    this->m_cipher->numRounds = rounds;
    this->m_cipher->ECRYPT_init();

    uint8_t keyBuff[DECIM_DEFAULT_KEY_LEN];
    uint8_t ivBuff[DECIM_DEFAULT_IV_LEN];
    for (unsigned char i = 0; i < DECIM_DEFAULT_KEY_LEN; i++) {
        keyBuff[i] = (uint8_t) this->m_gen->operator()();
    }
    for (unsigned char i = 0; i < DECIM_DEFAULT_IV_LEN; i++) {
        ivBuff[i] = (uint8_t) this->m_gen->operator()();
    }

    this->m_cipher->ECRYPT_keysetup(&this->m_ctx, keyBuff, (u32)DECIM_DEFAULT_KEY_LEN*8, (u32)DECIM_DEFAULT_IV_LEN*8);
    this->m_cipher->ECRYPT_ivsetup(&this->m_ctx, ivBuff);
}

DataSourceDecim::~DataSourceDecim() {
    if (this->m_gen != nullptr) {
        delete this->m_gen;
        this->m_gen = nullptr;
    }

    if (this->m_cipher != nullptr){
        delete this->m_cipher;
        this->m_cipher = nullptr;
    }
}

long long DataSourceDecim::getAvailableData() {
    return 9223372036854775807; // 2^63-1
}

void DataSourceDecim::read(char *buffer, size_t size) {
    this->m_cipher->DECIM_keystream_bytes(&this->m_ctx, (u8*)buffer, (u32)size);
}

std::string DataSourceDecim::desc() {
    std::stringstream ss;
    ss << "Decim-r" << this->m_rounds;
    return ss.str();
}
