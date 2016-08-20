#pragma once

#include <core/json.h>
#include <pcg/pcg_extras.hpp>
#include <random>
#include <sstream>

template <class Type> struct seed {
    using value_type = Type;

    const value_type value() const {
        return _value;
    }

    std::string to_string() const {
        std::ostringstream out;

        out.flags(std::ios_base::hex);
        out << _value;

        return out.str();
    }

    static seed create(std::string str) {
        value_type value;
        std::istringstream in{str};

        in.flags(std::ios_base::hex);
        in >> value;

        return seed(value);
    }

    static seed create(std::nullptr_t) {
        pcg_extras::seed_seq_from<std::random_device> seed_source;
        return seed(pcg_extras::generate_one<value_type>(seed_source));
    }

    static seed create(const core::json& settings) {
        if (settings.is_null())
            return seed::create(nullptr);
        else
            return seed::create(settings.get<std::string>());
    }

private:
    const value_type _value;

    explicit seed(const value_type value)
        : _value(value) {
    }
};
