#include "sha3_stream.h"
#include "sha3_factory.h"
#include "sha3_interface.h"
#include "eacirc/streams.h"
#include <algorithm>

template <typename I>
static void
hash_data(sha3_interface& hasher, const I& data, std::uint8_t* hash, const std::size_t hash_size) {
    using std::to_string;

    int status = hasher.Init(int(8 * hash_size));
    if (status != 0)
        throw std::runtime_error("cannot initialize hash (code: " + to_string(status) + ")");

    status = hasher.Update(&(*data.begin()), 8 * (data.end() - data.begin()));
    if (status != 0)
        throw std::runtime_error("cannot update the hash (code: " + to_string(status) + ")");

    status = hasher.Final(hash);
    if (status != 0)
        throw std::runtime_error("cannot finalize the hash (code: " + to_string(status) + ")");
}

sha3_stream::sha3_stream(const json& config, std::size_t osize)
    :  stream(osize)
    , _algorithm(config.at("algorithm").get<std::string>())
    , _round(config.at("round"))
    , _hash_size(std::size_t(config.at("hash-bitsize")))
    , _source(make_stream(config.at("source"), _hash_size / 8)) // TODO: hash-input-size?
    , _hasher(sha3_factory::create(_algorithm, unsigned(_round)))
    , _data(osize) {

    if ((std::size_t(config.at("hash-bitsize")) % 8) != 0)
        throw std::runtime_error("the SHA-3 hash-bitsize parameter must be multiple of 8");
    if ((_hash_size / 8) != osize)
        throw std::runtime_error("multiple/parts hashes in one vector are not supported yet");
}

sha3_stream::sha3_stream(sha3_stream&&) = default;
sha3_stream::~sha3_stream() = default;

vec_view sha3_stream::next() {
    vec_view view = _source->next();

    hash_data(*_hasher, view, _data.data(), _hash_size); // TODO: solve hash_size != osize

    return make_cview(_data);
}
