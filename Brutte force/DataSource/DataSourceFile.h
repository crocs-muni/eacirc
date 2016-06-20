//
// Created by Dusan Klinec on 19.06.16.
//

#ifndef BRUTTE_FORCE_DATASOURCEFILE_H
#define BRUTTE_FORCE_DATASOURCEFILE_H


#include <string>
#include <fstream>
#include "DataSource.h"

class DataSourceFile : public DataSource {
public:
    DataSourceFile(std::string fileName);
    virtual long long getAvailableData() override;
    virtual void read(char *buffer, size_t size) override;
    virtual std::string desc() override;

private:
    std::string fileName;
    long long fileSize;
    std::ifstream in;
};


#endif //BRUTTE_FORCE_DATASOURCEFILE_H
