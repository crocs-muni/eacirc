#pragma once

#include <core/dataset.h>

struct backend {
    virtual ~backend() = default;

    virtual void train(dataset const& a, dataset const& b) = 0;
    virtual double test(dataset const& a, dataset const& b) = 0;
    //virtual void dump_to_graph(const std::string &filename) = 0;
};
