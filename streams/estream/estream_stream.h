#pragma once

#include "estream_cipher.h"
#include <core/json.h>
#include <core/random.h>
#include <core/stream.h>

enum class estream_init_frequency { ONLY_ONCE, EVERY_VECTOR };

enum class estream_plaintext_type {
    ZEROS,
    ONES,
    RANDOM,
    BIASRANDOM,
    LUTRANDOM,
    COUNTER,
    FLIP5BITS,
    HALFBLOCKSAC
};

struct estream_stream : stream {
    estream_stream(const json& config, default_seed_source& seeder);

    void read(dataset& set) override;

protected:
    void setup_plaintext();

private:
    const estream_init_frequency _initfreq;
    const estream_plaintext_type _plaitext_type;

    polymorphic_generator _rng;
    counter _counter;
    std::vector<std::uint8_t> _plaintext;
    std::vector<std::uint8_t> _encrypted;

    estream_cipher _algorithm;
};
