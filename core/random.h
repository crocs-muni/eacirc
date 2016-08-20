#pragma once

#include <fstream>
#include <limits.h>
#include <pcg/pcg_extras.hpp>
#include <pcg/pcg_random.hpp>
#include <random>

namespace core {

using default_rng = pcg32;
using main_seed_source = pcg_extras::seed_seq_from<pcg32>;

template <class Type> struct qrng_engine {
    using result_type = Type;

    qrng_engine(const std::string file)
        : _file(file) {
        if (!_file.is_open())
            throw std::runtime_error("Can't open file \"" + file + "\".");
    }

    result_type operator()() {
        result_type value;
        _file.read(reinterpret_cast<std::ifstream::char_type*>(&value));
        if (_file.eof())
            throw std::range_error("reading of qrng data reached end of file.");
        if (_file.fail())
            throw std::runtime_error("an unrecoverable read error during reading of qrng data");
        return value;
    }

    static constexpr result_type min() {
        return std::numeric_limits<result_type>::min();
    }

    static constexpr result_type max() {
        return std::numeric_limits<result_type>::max();
    }

private:
    std::ifstream _file;
};

} // namespace core
