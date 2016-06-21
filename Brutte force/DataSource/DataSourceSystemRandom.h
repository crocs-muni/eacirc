//
// Created by Dusan Klinec on 21.06.16.
//

#ifndef BRUTTE_FORCE_DATASOURCESYSTEMRANDOM_H
#define BRUTTE_FORCE_DATASOURCESYSTEMRANDOM_H

#include "DataSource.h"
#include <random>

class DataSourceSystemRandom : public DataSource{
public:
    DataSourceSystemRandom(unsigned long seed = 0);
    ~DataSourceSystemRandom();

    virtual long long getAvailableData() override;
    virtual void read(char *buffer, size_t size) override;
    virtual std::string desc() override;

protected:
    std::minstd_rand * m_gen;
private:
};

#endif //BRUTTE_FORCE_DATASOURCESYSTEMRANDOM_H
