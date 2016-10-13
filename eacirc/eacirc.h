#pragma once

#include "backend.h"
#include "seed.h"
#include <core/json.h>
#include <core/stream.h>
#include <memory>

struct eacirc {
    eacirc(std::string cofig);

    eacirc(std::istream& config)
        : eacirc(json::parse(config)) {}

    eacirc(std::istream&& config)
        : eacirc(json::parse(config)) {}

    eacirc(json&& config)
        : eacirc(config) {}

    eacirc(json const& config);

    void run();

private:
    const json _config;
    const seed _seed;

    const std::uint64_t _num_of_epochs;
    const unsigned _significance_level;
    const unsigned _tv_size;
    const std::uint64_t _tv_count;

    std::unique_ptr<backend> _backend;
    std::unique_ptr<stream> _stream_a;
    std::unique_ptr<stream> _stream_b;
};
