//
// Created by Dusan Klinec on 21.06.16.
//

#ifndef BRUTTE_FORCE_DATASOURCEDECIM_H
#define BRUTTE_FORCE_DATASOURCEDECIM_H

#include <random>
#include "../DataGenerators/estream/EstreamInterface.h"
#include "../DataGenerators/estream/ciphers/decim/ecrypt-sync.h"
#include "DataSource.h"

class DataSourceDecim : public DataSource{
public:
    DataSourceDecim(unsigned long seed = 0, int rounds = 8);
    ~DataSourceDecim();

    virtual long long getAvailableData() override;
    virtual void read(char *buffer, size_t size) override;
    virtual std::string desc() override;

protected:
    std::minstd_rand * m_gen;
    DECIM_ctx m_ctx;
    ECRYPT_Decim * m_cipher;
    int m_rounds;
private:

};


#endif //BRUTTE_FORCE_DATASOURCEDECIM_H
