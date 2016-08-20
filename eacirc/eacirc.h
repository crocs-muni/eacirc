#pragma once

#include "backend.h"
#include "seed.h"
#include <core/json.h>
#include <core/random.h>
#include <core/stream.h>
#include <memory>

struct eacirc {
    eacirc(std::string config);

    eacirc(std::istream& config);
    eacirc(std::istream&& config)
        : eacirc(config) {
    }

    void run();

private:
    const core::json _config;

    const seed<std::uint64_t> _seed;
    core::main_seed_source _seed_source;

    const std::size_t _num_of_epochs;
    const std::size_t _significance_level;

    std::unique_ptr<backend> _backend;
    std::unique_ptr<core::stream> _stream_a;
    std::unique_ptr<core::stream> _stream_b;
};
