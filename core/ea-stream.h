#pragma once

#include "ea-iterators.h"
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ea {

template <class I> using tv_iterator = step_iterator<range_iterator<I>>;

struct dataset {
    using storage = std::vector<std::uint8_t>;

    using iterator = tv_iterator<storage::iterator>;
    using const_iterator = tv_iterator<storage::const_iterator>;

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

private:
    storage _data;
};

struct stream_error : std::logic_error {
    stream_error(std::string str)
        : logic_error(std::move(str)) {}
};

struct stream {
    virtual ~stream() = default;

    virtual void read(dataset &) = 0;
};

} // namespace ea
