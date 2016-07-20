#pragma once

#include "ea-iterators.h"
#include <cstdint>
#include <vector>

namespace ea {

template <class I> using tv_iterator = step_iterator<range_iterator<I>>;

struct dataset {
    using storage = std::vector<std::uint8_t>;

    using iterator = tv_iterator<storage::iterator>;
    using cone_iterator = tv_iterator<storage::const_iterator>;

private:
    storage _data;
};

struct datastream {
    virtual ~datastream() = default;

    virtual void read(dataset &) = 0;
};

} // namespace ea
