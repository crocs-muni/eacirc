//
// Created by Dusan Klinec on 21.06.16.
//

#include "DataSourceSystemRandom.h"
DataSourceSystemRandom::DataSourceSystemRandom(unsigned long seed) {
    this->gen = new std::minstd_rand((unsigned int) seed);
}

DataSourceSystemRandom::~DataSourceSystemRandom() {
    if (this->gen != nullptr) {
        delete this->gen;
        this->gen = nullptr;
    }
}

long long DataSourceSystemRandom::getAvailableData() {
    return 9223372036854775807; // 2^63-1
}

void DataSourceSystemRandom::read(char *buffer, size_t size) {
    for(size_t offset = 0; offset < size; offset += 1) {
        buffer[offset] = (uint8_t) this->gen->operator()();
    }
}

std::string DataSourceSystemRandom::desc() {
    return "SystemRandom";
}
