//
// Created by Dusan Klinec on 06.10.16.
//

#ifndef BRUTTE_FORCE_DATASOURCEESTREAM_H
#define BRUTTE_FORCE_DATASOURCEESTREAM_H

#include <random>
#include "DataSource.h"
#include "../DataGenerators/estream/EstreamInterface.h"

#define ESTREAM_DECIM 1
#define ESTREAM_TEA 2

#define ESTREAM_DEFAULT_IV_LEN 16
#define ESTREAM_DEFAULT_KEY_LEN 16

#define ESTREAM_ZERO_PLAINTEXT_BLOCK 8192

class DataSourceEstream : public DataSource {
public:
    DataSourceEstream(unsigned long seed, int function, int rounds);
    ~DataSourceEstream();

    virtual long long getAvailableData() override;
    virtual void read(char *buffer, size_t size) override;
    virtual void read(char *buffer, char* keys, char* messages, size_t size);
    virtual std::string desc() override;

protected:
    int m_function;
    int m_rounds;

    std::minstd_rand * m_gen;

    // Context + cipher
    void * m_ctx;
    EstreamInterface * m_estream;

    // If stream cipher, keystream is extracted by encrypting zero vector.
    // Otherwise for block cipher its CTR mode.
    bool m_stream_cipher;

    // Makes sense for a block cipher - size of an input block
    unsigned m_block_size_bytes;

    // Counter for block cipher CTR mode
    u64 m_counter;

    // Input block buffer for block ciphers (counter incrementation).
    u8 * m_input_block;

    // 00...0 array used as an input to the stream cipher to get the key stream
    static const unsigned char m_zero_plaintext[ESTREAM_ZERO_PLAINTEXT_BLOCK];

private:
};




#endif //BRUTTE_FORCE_DATASOURCEESTREAM_H
