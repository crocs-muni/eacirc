//
// Created by Dusan Klinec on 19.06.16.
//

#ifndef BRUTTE_FORCE_DATASOURCEAES_H
#define BRUTTE_FORCE_DATASOURCEAES_H

#include "DataSource.h"

class DataSourceAES : public DataSource {
public:
    DataSourceAES(unsigned seed);
    virtual long long getAvailableData() override;
    virtual void read(char *buffer, unsigned long long size) override;
    virtual std::string desc() override;
private:
};


#endif //BRUTTE_FORCE_DATASOURCEAES_H
