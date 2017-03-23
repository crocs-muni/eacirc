#pragma once

#include <core/json.h>
#include <core/stream.h>
#include <memory>

struct sha3_interface;

struct sha3_stream : stream {
    sha3_stream(const json& config);
    sha3_stream(sha3_stream&&);
    ~sha3_stream();

    void read(dataset& set) override;

private:
    const std::string _algorithm;
    const std::size_t _round;
    const std::size_t _hash_size;
    counter _counter;
    std::unique_ptr<sha3_interface> _hasher;
};
