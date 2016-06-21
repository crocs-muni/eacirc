//
// Created by Dusan Klinec on 21.06.16.
//

#ifndef BRUTTE_FORCE_DATASOURCERC4_H
#define BRUTTE_FORCE_DATASOURCERC4_H

#include "DataSource.h"
#include "../DataGenerators/arcfour.h"

class DataSourceRC4 : public DataSource {
public:
    DataSourceRC4(unsigned long seed = 0, unsigned keySize = 16);
    ~DataSourceRC4() {}

    virtual long long getAvailableData() override;
    virtual void read(char *buffer, size_t size) override;
    virtual std::string desc() override;

protected:
    BYTE m_state[RC4_STATE_SIZE];
    BYTE m_key[RC4_STATE_SIZE];
    unsigned m_keySize;
private:
};


#endif //BRUTTE_FORCE_DATASOURCERC4_H
