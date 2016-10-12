#include "streams.h"
#include <algorithm>
#include <core/json.h>
#include <core/memory.h>
#include <core/stream.h>
#include <fstream>
#include <limits>
#include <pcg/pcg_random.hpp>
#include <random>
#include <streams/sha3/sha3_stream.h>
#include <string>

namespace _impl {

    template <std::uint8_t value> struct const_stream : stream {
        void read_dataset(dataset& set) override {
            std::fill_n(set.rawdata(), set.rawsize(), value);
        }

        void read_raw(std::basic_ostream<value_type>& os, std::size_t size) override {
            std::fill_n(std::ostreambuf_iterator<value_type>(os), size, value);
        }
    };

    template <typename Generator> struct rng_stream : stream {
        template <typename Seeder>
        rng_stream(Seeder&& seeder)
            : _rng(std::forward<Seeder>(seeder)) {}

        void read_dataset(dataset& set) override {
            std::generate_n(set.rawdata(), set.rawsize(), [this]() {
                return std::uniform_int_distribution<value_type>()(_rng);
            });
        }

        void read_raw(std::basic_ostream<value_type>& os, std::size_t size) override {
            std::generate_n(std::ostreambuf_iterator<value_type>(os), size, [this]() {
                return std::uniform_int_distribution<value_type>()(_rng);
            });
        }

    private:
        Generator _rng;
    };

} // namespace _impl

/**
 * \brief Stream of true bits
 */
using true_stream = _impl::const_stream<std::numeric_limits<std::uint8_t>::max()>;

/**
 * \brief Stream of false bits
 */
using false_stream = _impl::const_stream<std::numeric_limits<std::uint8_t>::min()>;

/**
 * \brief Stream of data produced by Merseine Twister
 */
using mt19937_stream = _impl::rng_stream<std::mt19937>;

/**
 * \brief Stream of data produced by PCG (Permutation Congruential Generator)
 */
using pcg32_stream = _impl::rng_stream<pcg32>;

/**
 * @brief Stream of data read from a file
 */
struct file_stream : stream {
    file_stream(const json& config)
        : _path(config.at("path").get<std::string>())
        , _istream(_path, std::ios::binary) {}

    void read_raw(std::basic_ostream<value_type>&, std::size_t) override {
        throw std::logic_error("this function is useless for file stream");
    }

    void read_dataset(dataset& set) override {
        _istream.read(set.rawdata(), std::streamsize(set.rawsize()));

        if (_istream.fail())
            throw std::runtime_error("I/O error while reading a file " + _path);
        if (_istream.eof())
            throw std::runtime_error("end of file " + _path + " reached, not enough data!");
    }

private:
    const std::string _path;
    std::basic_ifstream<std::uint8_t> _istream;
};

std::unique_ptr<stream> make_stream(json const& config, default_seed_source& seeder) {
    std::string type = config.at("type");

    if (type == "file-stream")
        return std::make_unique<file_stream>(config);
    else if (type == "true-stream")
        return std::make_unique<true_stream>();
    else if (type == "false-stream")
        return std::make_unique<true_stream>();
    else if (type == "mt19937-stream")
        return std::make_unique<mt19937_stream>(seeder);
    else if (type == "pcg32-stream")
        return std::make_unique<pcg32_stream>(seeder);
    else if (type == "sha3-stream")
        return std::make_unique<sha3_stream>(config);
    else
        throw std::runtime_error("requested stream named \"" + type + "\" does not exist");
}
