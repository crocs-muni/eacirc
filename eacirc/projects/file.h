#pragma once

#include <core/exceptions.h>
#include <core/logger.h>
#include <core/project.h>
#include <fstream>
#include <istream>

template <size_t Size>
std::basic_istream<u8>& operator>>(std::basic_istream<u8>& is, DataVecStorage<Size>& tvs) {
    is.read(tvs.front().data(), Size * tvs.size());

    if (!is.good()) {
        Logger::error() << "An error occurred during read of test vectors from the stream."
                        << std::endl;
        throw fatal_error();
    }
    return is;
}

template <size_t Size> class FileStream : public TestedStream {
    std::basic_ifstream<u8> _ifs;

public:
    FileStream(const std::string file) : _ifs(file, std::ios::binary) {}

    void read(DataVectors& tvs) override { _ifs >> dynamic_cast<DataVecStorage<Size>&>(tvs); }
};
