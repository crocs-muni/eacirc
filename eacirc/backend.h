#pragma once

#include <core/stream.h>

struct backend {
    virtual ~backend() = default;

    virtual double train() = 0;

protected:
    backend(core::stream& stream_a, core::stream& stream_b)
        : _stream_a(stream_a)
        , _stream_b(stream_b) {
    }

    core::stream& stream_a() {
        return _stream_a;
    }

    core::stream& stream_b() {
        return _stream_b;
    }

    core::stream const& stream_a() const {
        return _stream_a;
    }

    core::stream const& stream_b() const {
        return _stream_b;
    }

private:
    core::stream& _stream_a;
    core::stream& _stream_b;
};
