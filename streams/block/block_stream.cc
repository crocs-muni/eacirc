#include "block_stream.h"
#include "eacirc/streams.h"
#include "block_factory.h"
#include "block_interface.h"

static std::size_t compute_vector_size(const std::size_t block_size, const std::size_t osize) {
    return (block_size > osize)
            ? block_size
            : (block_size % osize) ? ((osize / block_size) + 1) * block_size
                                   : osize
}

block_stream::block_stream(const json& config, std::size_t osize)
    : stream(osize)
    , _algorithm(config.at("algorithm").get<std::string>())
    , _round(config.at("round"))
    , _block_size(std::size_t(config.at("block-size")))
    , _source(make_stream(config.at("source"), _block_size))
    , _iv(make_stream(config.at("iv"), _block_size))
    , _key(make_stream(config.at("key"), _block_size)) // TODO: check, if key_len is always of _block_size
    , _encryptor(block_factory::make_cipher(_algorithm, unsigned(_round)))
    , _decryptor(block_factory::make_cipher(_algorithm, unsigned(_round)))
    , _data(compute_vector_size(_block_size, osize))
{
    vec_view iv_view = _iv->next();
    _encryptor->ECRYPT_ivsetup(iv_view.data());
    _decryptor->ECRYPT_ivsetup(iv_view.data());

    vec_view key_view = _key->next();
    _encryptor->ECRYPT_keysetup(key_view.data(), 8 * _block_size, 8 * _block_size);
    _decryptor->ECRYPT_keysetup(key_view.data(), 8 * _block_size, 8 * _block_size);
}

block_stream::block_stream(block_stream&&) = default;
block_stream::~block_stream() = default;

vec_view block_stream::next() {

    // TODO: reinit key for every vector: does it make sense?
    // if (_b_reinit_every_tv) {
    //    vec_view key_view = _key->next();
    //    _encryptor->ECRYPT_keysetup(key_view.data(), 8 * _block_size, 8 * _block_size);
    //    _decryptor->ECRYPT_keysetup(key_view.data(), 8 * _block_size, 8 * _block_size);
    // }

    for (auto beg = _data.begin(); beg != _data.end(); beg += _block_size) {
        vec_view view = _source->next();

        // BEWARE: only able to proccess max 2GB of plaintext
        _encryptor->ECRYPT_encrypt_bytes(view.data(), &(*beg), u32(_block_size));
    }

    return make_view(_data.cbegin(), osize());
}
