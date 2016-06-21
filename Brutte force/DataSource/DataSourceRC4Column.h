//
// Created by Dusan Klinec on 21.06.16.
//

#ifndef BRUTTE_FORCE_DATASOURCERC4COLUMN_H
#define BRUTTE_FORCE_DATASOURCERC4COLUMN_H

#include "DataSource.h"
#include "../DataGenerators/arcfour.h"
#include <random>

class DataSourceRC4Column : public DataSource{
public:
    DataSourceRC4Column(unsigned long seed = 0, unsigned blockSize = 16, unsigned keySize = 16);
    ~DataSourceRC4Column();

    virtual long long getAvailableData() override;
    virtual void read(char *buffer, size_t size) override;
    virtual std::string desc() override;

protected:
    std::minstd_rand * gen;
    unsigned blockSize;
    unsigned keySize;
private:
};


#endif //BRUTTE_FORCE_DATASOURCERC4COLUMN_H
