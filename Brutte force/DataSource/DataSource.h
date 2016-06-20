//
// Created by Dusan Klinec on 19.06.16.
//

#ifndef BRUTTE_FORCE_DATASOURCE_H
#define BRUTTE_FORCE_DATASOURCE_H

#include <string>

class DataSource {
public:
    /**
     * Number of available bytes in this data source.
     */
    virtual long long getAvailableData() = 0;

    /**
     * Reads size data to the buffer.
     */
    virtual void read (char *buffer, size_t size) = 0;

    /**
     * Describe the data source.
     */
    virtual std::string desc() = 0;
};


#endif //BRUTTE_FORCE_DATASOURCE_H
