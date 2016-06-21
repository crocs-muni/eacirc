//
// Created by Dusan Klinec on 21.06.16.
//

#ifndef BRUTTE_FORCE_DATASOURCEDECIMCOLUMN_H
#define BRUTTE_FORCE_DATASOURCEDECIMCOLUMN_H

#include <random>
#include "../DataGenerators/estream/EstreamInterface.h"
#include "../DataGenerators/estream/ciphers/decim/ecrypt-sync.h"
#include "DataSource.h"

class DataSourceDecimColumn : public DataSource {
public:
    DataSourceDecimColumn(unsigned long seed = 0, int rounds = 8, unsigned blockSize = 16);
    ~DataSourceDecimColumn();
    virtual long long getAvailableData() override;
    virtual void read(char *buffer, size_t size) override;
    virtual std::string desc() override;

protected:
    std::minstd_rand *m_gen;
    DECIM_ctx m_ctx;
    ECRYPT_Decim *m_cipher;
    int m_rounds;
    unsigned m_blockSize;
private:

};



#endif //BRUTTE_FORCE_DATASOURCEDECIMCOLUMN_H
