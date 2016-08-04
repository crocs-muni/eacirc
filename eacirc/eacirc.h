#pragma once

#include <core/stream.h>
#include <memory>

struct eacirc {
    void run();

private:
    std::unique_ptr<core::stream> _stream_a;
    std::unique_ptr<core::stream> _stream_b;
};
