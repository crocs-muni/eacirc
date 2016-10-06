//
// Created by Dusan Klinec on 06.10.16.
//

#include "DataSourceEstream.h"
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <cassert>

const unsigned char DataSourceEstream::m_zero_plaintext[ESTREAM_ZERO_PLAINTEXT_BLOCK] = {0};

DataSourceEstream::DataSourceEstream(unsigned long seed, int function, int rounds) {
    // Safe initialization
    m_stream_cipher = true;
    m_block_size_bytes = 0;
    m_input_block = nullptr;
    m_ctx = nullptr;

    // Per-cipher initialization
    if (function == ESTREAM_DECIM){
        m_estream = new ECRYPT_Decim();
        m_ctx = malloc(sizeof(DECIM_ctx));

    } else if (function == ESTREAM_TEA) {
        m_estream = new ECRYPT_TEA();
        m_ctx = malloc(sizeof(TEA_ctx));
        m_stream_cipher = false;
        m_block_size_bytes = 8;

    } else {
        throw std::out_of_range("Unknown Estream function");

    }

    m_gen = new std::minstd_rand((unsigned int) seed);
    m_estream->numRounds = rounds;
    m_estream->ECRYPT_init();

    uint8_t keyBuff[ESTREAM_DEFAULT_KEY_LEN];
    uint8_t ivBuff[ESTREAM_DEFAULT_IV_LEN];
    for (unsigned char i = 0; i < ESTREAM_DEFAULT_KEY_LEN; i++) {
        keyBuff[i] = (uint8_t) this->m_gen->operator()();
    }
    for (unsigned char i = 0; i < ESTREAM_DEFAULT_IV_LEN; i++) {
        ivBuff[i] = (uint8_t) this->m_gen->operator()();
    }

    m_estream->ECRYPT_keysetup(this->m_ctx, keyBuff, (u32)ESTREAM_DEFAULT_KEY_LEN*8, (u32)ESTREAM_DEFAULT_IV_LEN*8);
    m_estream->ECRYPT_ivsetup(this->m_ctx, ivBuff);

    m_function = function;
    m_rounds = rounds;
    m_counter = m_gen->operator()();

    if (!m_stream_cipher && m_block_size_bytes > 0){
        assert(sizeof(u8) == 1);
        m_input_block = (u8*)malloc(m_block_size_bytes*sizeof(u8));
    }
}

DataSourceEstream::~DataSourceEstream() {
    if (this->m_gen != nullptr) {
        delete this->m_gen;
        this->m_gen = nullptr;
    }

    if (this->m_estream != nullptr){
        delete this->m_estream;
        this->m_estream = nullptr;
    }

    if (this->m_ctx != nullptr){
        free(this->m_ctx);
        this->m_ctx = nullptr;
    }

    if (this->m_input_block != nullptr){
        free(this->m_input_block);
        this->m_input_block = nullptr;
    }
}

long long DataSourceEstream::getAvailableData() {
    return 9223372036854775807; // 2^63-1
}

void DataSourceEstream::read(char *buffer, size_t size) {
    if (m_stream_cipher) {
        // Stream cipher -> extract key stream by encrypting zero vector
        for (size_t offset = 0; offset < size; offset += ESTREAM_ZERO_PLAINTEXT_BLOCK) {
            const size_t to_enc = std::min((size_t) ESTREAM_ZERO_PLAINTEXT_BLOCK, size - offset);
            this->m_estream->ECRYPT_encrypt_bytes(m_ctx, m_zero_plaintext, ((u8 *) (buffer)) + offset, (u32) to_enc);
        }

    } else {
        // Block cipher - encrypt counter.
        assert(m_input_block != NULL && m_block_size_bytes > 0);
        for(size_t offset = 0; offset < size; offset += m_block_size_bytes){
            memset(m_input_block, 0, m_block_size_bytes);
            memcpy(m_input_block, &m_counter, sizeof(m_counter));

            const size_t to_enc = std::min((size_t) m_block_size_bytes, size - offset);
            this->m_estream->ECRYPT_encrypt_bytes(m_ctx, m_input_block, ((u8 *) (buffer)) + offset, to_enc);
            ++m_counter;
        }
    }
}

std::string DataSourceEstream::desc() {
    std::stringstream ss;
    ss << "Estream-ID" << m_function << "-r" << m_rounds;
    return ss.str();
}

