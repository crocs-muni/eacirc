#include "sha3_stream.h"
#include "sha3_factory.h"
#include "sha3_interface.h"
#include <algorithm>

template <typename I, typename O>
static void hash_data(sha3_interface& hasher, const I& data, O& hash) {
    using std::to_string;

    int status = hasher.Init(int(8 * hash.size()));
    if (status != 0)
        throw std::runtime_error("cannot initialize hash (code: " + to_string(status) + ")");

    status = hasher.Update(data.data(), 8 * data.size());
    if (status != 0)
        throw std::runtime_error("cannot update the hash (code: " + to_string(status) + ")");

    status = hasher.Final(hash.data());
    if (status != 0)
        throw std::runtime_error("cannot finalize the hash (code: " + to_string(status) + ")");
}

sha3_stream::sha3_stream(const json& config)
    : block_stream(std::size_t(config.at("hash-bitsize")) / 8)
    , _algorithm(config.at("algorithm").get<std::string>())
    , _round(config.at("round"))
    , _hash_bitsize(config.at("hash-bitsize"))
    , _counter(8)
    , _hasher(sha3_factory::create(_algorithm, unsigned(_round))) {
    if ((_hash_bitsize % 8) != 0)
        throw std::runtime_error("the SHA-3 hash-bitsize parameter must be multiple of 8");
}

sha3_stream::sha3_stream(sha3_stream&&) = default;
sha3_stream::~sha3_stream() = default;

void sha3_stream::generate() {
    hash_data(*_hasher, _counter, block());
    _counter.increment();
}
