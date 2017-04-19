#pragma once

#include <eacirc-core/dataset.h>

struct backend {
    virtual ~backend() = default;

    virtual void train(dataset const& a, dataset const& b) = 0;
    virtual void set_produce_file(std::string filename, bool value) = 0;
    virtual double test(dataset const& a, dataset const& b) = 0;
};
