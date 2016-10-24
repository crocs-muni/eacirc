//
// Created by Dusan Klinec on 19.06.16.
//

#ifndef BRUTTE_FORCE_DATASOURCEMD5_H
#define BRUTTE_FORCE_DATASOURCEMD5_H


#include "DataSource.h"
#include "../DataGenerators/md5.h"

class DataSourceMD5 : public DataSource{
public:
    DataSourceMD5(unsigned long seed = 0, int Nr = 64);
    ~DataSourceMD5() {}

    virtual long long getAvailableData() override;
    virtual void read(char *buffer, size_t size) override;
    virtual std::string desc() override;
    virtual void read(char *buffer, char* messages, size_t size);

protected:
    MD5_DIGEST m_md5Accumulator; // accumulator for MD5
    int updateAccumulator();
    int m_rounds;
private:
};


#endif //BRUTTE_FORCE_DATASOURCEMD5_H
