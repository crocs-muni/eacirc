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
    virtual std::string desc() override;

protected:
    int m_function;
    int m_rounds;
    long long m_counter;

    void * m_ctx;
    std::minstd_rand * m_gen;
    EstreamInterface * m_estream;

    // 00...0 array used as an input to the estream to get the key stream
    static const unsigned char m_zero_plaintext[ESTREAM_ZERO_PLAINTEXT_BLOCK];

private:
};




#endif //BRUTTE_FORCE_DATASOURCEESTREAM_H
