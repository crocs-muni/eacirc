#include "sha3.h"
#include "sha3_interface.h"

namespace sha3 {

    static std::string to_string(sha3_mode mode) {
        switch (mode) {
        case sha3_mode::COUNTER:
            return "counter";
        }
    }

    static sha3_mode mode_from_string(std::string str) {
        if (str == to_string(sha3_mode::COUNTER))
            return sha3_mode::COUNTER;
        throw std::runtime_error("no such SHA-3 mode named \"" + str + "\" exits");
    }

    sha3_stream::sha3_stream(json const& config)
        : _mode(mode_from_string(config.at("mode")))
        , _algortihm(algorithm_from_string(config.at("algorithm")))
        , _round(config.at("round"))
        , _hash_bitsize(config.at("hash-bitsize"))
        , _counter()
        , _hasher(create_algorithm(_algortihm, _round)) {
        if ((_hash_bitsize % 8) != 0)
            throw std::runtime_error("the SHA-3 hash-bitsize parameter sould be multiple of 8");
    }

    sha3_stream::sha3_stream(sha3_stream&&) = default;

    sha3_stream::~sha3_stream() = default;

    void sha3_stream::read(byte_view out) {
        switch (_mode) {
        case sha3_mode::COUNTER:
            _mode_counter(out);
            break;
        }
    }

    void sha3_stream::_mode_counter(byte_view out) {
        auto beg = out.begin();
        auto end = out.end();

        const auto hash_size = _hash_bitsize / 8;
        const auto plain_bitsize = _counter.size() * 8;

        ASSERT_ALLWAYS((out.size() % hash_size) == 0);

        for (; beg != end; beg += hash_size) {
            _hash(_counter.data(), plain_bitsize, beg);
            ++_counter;
        }
    }

    void sha3_stream::_hash(std::uint8_t const* plaintext,
                            std::size_t plaintext_bitsize,
                            std::uint8_t* cyphertext) {
        int status = 0;

        status = _hasher->Init(int(_hash_bitsize));
        if (status != 0)
            throw stream_error("cant't initialize hash (status: " + std::to_string(status) + ")");

        status = _hasher->Update(plaintext, plaintext_bitsize);
        if (status != 0)
            throw stream_error("cant't update hash (status: " + std::to_string(status) + ")");

        status = _hasher->Final(cyphertext);
        if (status != 0)
            throw stream_error("cant't finalize hash (status: " + std::to_string(status) + ")");
    }

} // namespace sha3
