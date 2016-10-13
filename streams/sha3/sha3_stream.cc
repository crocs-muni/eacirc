#include "sha3_stream.h"
#include "sha3_factory.h"
#include "sha3_interface.h"
#include <algorithm>

template <typename I>
static void
hash_data(sha3_interface& hasher, const I& data, std::uint8_t* hash, std::size_t hash_size) {
    using std::to_string;

    int status = hasher.Init(int(8 * hash_size));
    if (status != 0)
        throw std::runtime_error("cannot initialize hash (code: " + to_string(status) + ")");

    status = hasher.Update(data.data(), 8 * data.size());
    if (status != 0)
        throw std::runtime_error("cannot update the hash (code: " + to_string(status) + ")");

    status = hasher.Final(hash);
    if (status != 0)
        throw std::runtime_error("cannot finalize the hash (code: " + to_string(status) + ")");
}

sha3_stream::sha3_stream(const json& config)
    : _algorithm(config.at("algorithm").get<std::string>())
    , _round(config.at("round"))
    , _hash_size(std::size_t(config.at("hash-bitsize")) / 8)
    , _counter(8)
    , _hasher(sha3_factory::create(_algorithm, unsigned(_round))) {
    if ((std::size_t(config.at("hash-bitsize")) % 8) != 0)
        throw std::runtime_error("the SHA-3 hash-bitsize parameter must be multiple of 8");
}

sha3_stream::sha3_stream(sha3_stream&&) = default;
sha3_stream::~sha3_stream() = default;

void sha3_stream::read(dataset& set) {
    auto beg = set.rawdata();
    auto end = set.rawdata() + set.rawsize();

    if ((set.rawsize() % _hash_size) != 0)
        throw std::runtime_error("SHA-3 dataset size must be multiple of hash size");

    for (; beg != end; beg += _hash_size) {
        hash_data(*_hasher, _counter, beg, _hash_size);
        _counter.increment();
    }
}
