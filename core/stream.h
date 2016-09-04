#pragma once

#include "dataset.h"
#include <stdexcept>

struct stream_error : std::runtime_error {
    stream_error(std::string const& what)
        : runtime_error(what) {}

    stream_error(char const* what)
        : runtime_error(what) {}
};

struct stream {
    virtual ~stream() = default;
    virtual void read(dataset& data) = 0;
};
