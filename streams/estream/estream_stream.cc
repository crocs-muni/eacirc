#include "estream_stream.h"

static estream_init_frequency create_init_frequency(const std::string& frequency) {
    // clang-format off
    if (frequency == "only-once")    return estream_init_frequency::ONLY_ONCE;
    if (frequency == "every-vector") return estream_init_frequency::EVERY_VECTOR;
    // clang-format on

    throw std::runtime_error("requested eSTREAM initialization frequency named \"" + frequency +
                             "\" does not exist");
}

static estream_plaintext_type create_plaintext_type(const std::string& type) {
    // clang-format off
    if (type == "zeros")          return estream_plaintext_type::ZEROS;
    if (type == "ones")           return estream_plaintext_type::ONES;
    if (type == "random")         return estream_plaintext_type::RANDOM;
    if (type == "biasrandom")     return estream_plaintext_type::BIASRANDOM;
    if (type == "lutrandom")      return estream_plaintext_type::LUTRANDOM;
    if (type == "counter")        return estream_plaintext_type::COUNTER;
    if (type == "flip5bits")      return estream_plaintext_type::FLIP5BITS;
    if (type == "half-block-SAC") return estream_plaintext_type::HALFBLOCKSAC;
    // clang-format on

    throw std::runtime_error("requested eSTREAM plaintext type name \"" + type +
                             "\" is not available");
}

estream_stream::estream_stream(const json& config, default_seed_source& seeder)
    : _initfreq(create_init_frequency(config.at("init-frequency")))
    , _plaitext_type(create_plaintext_type(config.at("plaintext-type")))
    , _rng(config.at("generator").get<std::string>(), seeder)
    , _counter(estream_cipher::block_size)
    , _plaintext(estream_cipher::block_size)
    , _encrypted(estream_cipher::block_size)
    , _algorithm(config.at("algorithm"),
                 config.at("round").is_null() ? optional<unsigned>{nullopt}
                                              : optional<unsigned>{unsigned(config.at("round"))},
                 config.at("iv-type"),
                 config.at("key-type"),
                 config.count("heatmap") != 0 ? std::uint64_t(config.at("heatmap")) : 0x0u) {
    if (_initfreq == estream_init_frequency::ONLY_ONCE) {
        _algorithm.setup_key(_rng);
        _algorithm.setup_iv(_rng);
    }
}

void estream_stream::setup_plaintext() {
    switch (_plaitext_type) {
    case estream_plaintext_type::ZEROS:
        std::fill(_plaintext.begin(), _plaintext.end(), 0x00u);
        break;
    case estream_plaintext_type::ONES:
        std::fill(_plaintext.begin(), _plaintext.end(), 0xffu);
        break;
    case estream_plaintext_type::RANDOM:
        std::generate(_plaintext.begin(), _plaintext.end(), _rng);
        break;
    case estream_plaintext_type::BIASRANDOM:
    case estream_plaintext_type::LUTRANDOM:
        throw std::logic_error("feature not yet implemented");
    case estream_plaintext_type::COUNTER:
        std::copy(_counter.begin(), _counter.end(), _plaintext.begin());
        _counter.increment();
        break;
    case estream_plaintext_type::FLIP5BITS:
    case estream_plaintext_type::HALFBLOCKSAC:
        throw std::logic_error("feature not yet implemented");
    }
}

void estream_stream::read(dataset& set) {
    auto beg = set.rawdata();
    auto end = set.rawdata() + set.rawsize();

    if ((set.rawsize() % estream_cipher::block_size) != 0)
        throw std::runtime_error("eSTREAM dataset size must be multiple of block size");

    for (; beg != end; beg += estream_cipher::block_size) {
        setup_plaintext();

        if (_initfreq == estream_init_frequency::EVERY_VECTOR) {
            _algorithm.setup_key(_rng);
            _algorithm.setup_iv(_rng);
        }

        _algorithm.encrypt(_plaintext.data(), _encrypted.data(), _plaintext.size());

        std::copy(_encrypted.begin(), _encrypted.end(), beg);
    }
}
