#include "sha3_stream.h"
#include "eacirc/streams.h"
#include "sha3_factory.h"
#include "sha3_interface.h"
#include <algorithm>

template <typename I>
static void
hash_data(sha3_interface& hasher, const I& data, std::uint8_t* hash, const std::size_t hash_size) {
    using std::to_string;

    int status = hasher.Init(int(hash_size * 8));
    if (status != 0)
        throw std::runtime_error("cannot initialize hash (code: " + to_string(status) + ")");

    status = hasher.Update(&(*data.begin()), 8 * (data.end() - data.begin()));
    if (status != 0)
        throw std::runtime_error("cannot update the hash (code: " + to_string(status) + ")");

    status = hasher.Final(hash);
    if (status != 0)
        throw std::runtime_error("cannot finalize the hash (code: " + to_string(status) + ")");
}

sha3_stream::sha3_stream(const json& config, default_seed_source &seeder, std::size_t osize)
    : stream(osize)
    , _algorithm(config.at("algorithm").get<std::string>())
    , _round(config.at("round"))
    , _hash_size(std::size_t(config.at("hash-bitsize")) / 8)
    , _source(make_stream(config.at("source"), seeder, _hash_size)) // TODO: hash-input-size?
    , _hasher(sha3_factory::create(_algorithm, unsigned(_round)))
    , _data((_hash_size > osize)
                    ? _hash_size
                    : (_hash_size % osize) ? ((osize / _hash_size) + 1) * _hash_size
                                           : osize) { // round osize to multiple of _hash_size

    if ((std::size_t(config.at("hash-bitsize")) % 8) != 0)
        throw std::runtime_error("the SHA-3 hash-bitsize parameter must be multiple of 8");
}

sha3_stream::sha3_stream(sha3_stream&&) = default;
sha3_stream::~sha3_stream() = default;

vec_view sha3_stream::next() {

    for (auto beg = _data.begin(); beg != _data.end(); beg += _hash_size) {
        vec_view view = _source->next();

        hash_data(*_hasher, view, &(*beg), _hash_size);
    }

    return make_view(_data.cbegin(), osize());
}
