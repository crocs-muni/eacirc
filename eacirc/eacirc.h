#pragma once

#include "backend.h"
#include <core/project.h>
#include <memory>
#include <string>
#include <random>

struct Eacirc {
private:
    std::mt19937 _main_generator;

    std::unique_ptr<Stream> _stream_A;
    std::unique_ptr<Stream> _stream_B;
    std::unique_ptr<Backend> _backend;

    unsigned _tv_size;
    unsigned _num_of_tvs;
    unsigned _num_of_epochs;
    unsigned _change_frequency;
    unsigned _significance_level;

public:
    Eacirc(const std::string config);

    void run();
};
