//
// Created by Dusan Klinec on 27.06.16.
//

#include "DataSourceSHA3.h"

#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include "../DataGenerators/sha3/hash_functions/Tangle/Tangle_sha3.h"
#include "../DataGenerators/sha3/hash_functions/MD6/MD6_sha3.h"
#include "../DataGenerators/sha3/hash_functions/Keccak/Keccak_sha3.h"
#include "../DataGenerators/md5.h"

DataSourceSHA3::DataSourceSHA3(unsigned long seed, int hash, int rounds, unsigned outputSize) {
    std::minstd_rand systemGenerator((unsigned int) seed);
    if (outputSize > 256){
        throw std::out_of_range("Maximum output size is 256B");
    }

    if (hash == SHA3_TANGLE){
        m_sha3 = new Tangle(rounds);

    } else if (hash == SHA3_MD6) {
        m_sha3 = new MD6(rounds);

    } else if (hash == SHA3_KECCAK) {
        m_sha3 = new Keccak;

    } else {
        throw std::out_of_range("Unknown SHA3 function");
    }

    m_hashFunction = hash;
    m_rounds = rounds;
    m_outputSize = outputSize;
    m_counter = systemGenerator.operator()();
}

DataSourceSHA3::~DataSourceSHA3() {
    if (this->m_sha3 != nullptr) {
        delete this->m_sha3;
        this->m_sha3 = nullptr;
    }
}

long long DataSourceSHA3::getAvailableData() {
    return 9223372036854775807; // 2^63-1
}

void DataSourceSHA3::read(char *buffer, size_t size) {
    BitSequence results[4096];
    for(size_t offset = 0; offset < size; offset += m_outputSize){
        m_sha3->Init(m_outputSize * 8);
        m_sha3->Update((const BitSequence *) &m_counter, sizeof(m_counter)*8);
        m_sha3->Final(results);
        m_counter += 1;
        memcpy(buffer+offset, results, std::min((size_t)m_outputSize, size-offset));
    }
}

std::string DataSourceSHA3::desc() {
    std::stringstream ss;
    ss << "SHA3-ID" << m_hashFunction << "-r" << m_rounds << "-out" << m_outputSize;
    return ss.str();
}

