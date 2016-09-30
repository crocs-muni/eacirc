#pragma once

#include "factory.h"
#include <core/iterators.h>
#include <core/json.h>
#include <core/stream.h>

struct sha3_interface;

namespace sha3 {

    enum class sha3_mode { COUNTER };

    struct sha3_stream : stream {
        sha3_stream(json const& config);

        sha3_stream(sha3_stream&&);

        ~sha3_stream();

        void read(byte_view out) override;

        std::size_t output_block_size() const override { return _hash_bitsize; }

    private:
        const sha3_mode _mode;
        const sha3_algorithm _algortihm;
        const unsigned _round;
        const unsigned _hash_bitsize;

        counter _counter;
        std::unique_ptr<sha3_interface> _hasher;

        void _mode_counter(byte_view out);

        void _hash(std::uint8_t const* plaintext,
                   std::size_t plaintext_bitsize,
                   std::uint8_t* cyphertext);
    };

} // namespace sha3
