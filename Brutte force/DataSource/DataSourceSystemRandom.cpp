//
// Created by Dusan Klinec on 21.06.16.
//

#include "DataSourceSystemRandom.h"
DataSourceSystemRandom::DataSourceSystemRandom(unsigned long seed) {
    this->m_gen = new std::minstd_rand((unsigned int) seed);
}

DataSourceSystemRandom::~DataSourceSystemRandom() {
    if (this->m_gen != nullptr) {
        delete this->m_gen;
        this->m_gen = nullptr;
    }
}

long long DataSourceSystemRandom::getAvailableData() {
    return 9223372036854775807; // 2^63-1
}

void DataSourceSystemRandom::read(char *buffer, size_t size) {
    for(size_t offset = 0; offset < size; offset += 1) {
        buffer[offset] = (uint8_t) this->m_gen->operator()();
    }
}

std::string DataSourceSystemRandom::desc() {
    return "SystemRandom";
}
