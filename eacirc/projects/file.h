#pragma once

#include <core/exceptions.h>
#include <core/logger.h>
#include <core/project.h>
#include <fstream>

struct FileStream final : Stream {
    std::ifstream _ifs;

public:
    FileStream(const std::string file) : _ifs(file, std::ifstream::binary) {
        if (!_ifs.is_open()) {
            Logger::error() << "Cannot open file " << file << std::endl;
            throw fatal_error();
        }
    }

    void read(Dataset& tvs) override {
        _ifs.read(
                reinterpret_cast<char*>(tvs.data()),
                static_cast<std::streamsize>(tvs.num_of_tvs() * tvs.tv_size()));

        if (!_ifs.good()) {
            Logger::error() << "An error occurred during read of test vectors from the stream."
                            << std::endl;
            Logger::error() << _ifs.fail() << std::endl;
            Logger::error() << _ifs.bad() << std::endl;
            Logger::error() << _ifs.eof() << std::endl;
            throw fatal_error();
        }
    }
};
