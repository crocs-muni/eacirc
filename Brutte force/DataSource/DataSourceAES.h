//
// Created by Dusan Klinec on 19.06.16.
//

#ifndef BRUTTE_FORCE_DATASOURCEAES_H
#define BRUTTE_FORCE_DATASOURCEAES_H

#include "DataSource.h"
#include "../dynamic_bitset.h"

class DataSourceAES : public DataSource {
public:
    DataSourceAES(unsigned long seed = 0, int rounds = 10);
    ~DataSourceAES() {}


    virtual long long getAvailableData() override;
    virtual void read(char *buffer, size_t size) override;
    virtual std::string desc() override;
    virtual void read(char *buffer, char* key, char* messages, size_t size);

private:
    uint8_t m_key[16];
    uint8_t m_iv[16];
    int m_rounds;
    u64 m_counter;
private:
};


#endif //BRUTTE_FORCE_DATASOURCEAES_H
