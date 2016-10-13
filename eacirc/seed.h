#pragma once

#include <core/json.h>
#include <cstdint>
#include <string>

struct seed {
    using value_type = std::uint64_t;

    explicit seed(std::nullptr_t);
    explicit seed(std::string const str);

    static seed create(json const& object) {
        if (object.is_null())
            return seed(nullptr);
        else
            return seed(object.get<std::string>());
    }

    operator std::string() const;
    operator value_type() const { return _value; }

private:
    const value_type _value;
};
