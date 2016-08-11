#pragma once

#include "seed.h"
#include <core/json.h>
#include <core/stream.h>
#include <memory>

struct eacirc {
    eacirc(std::string settings);

    eacirc(std::istream &settings);
    eacirc(std::istream &&settings)
        : eacirc(settings) {}

    void run();

private:
    const core::json _settings;
    const seed<std::uint64_t> _seed;

    std::unique_ptr<core::stream> _stream_a;
    std::unique_ptr<core::stream> _stream_b;
};
