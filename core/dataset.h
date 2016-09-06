#pragma once

#include "iterators.h"
#include <vector>

struct dataset {
    using storage = std::vector<std::uint8_t>;

    using iterator = step_iterator<view_iterator<typename storage::iterator>>;
    using const_iterator = step_iterator<view_iterator<typename storage::const_iterator>>;

    dataset()
        : _m(0)
        , _data() {}

    dataset(unsigned m, std::size_t n)
        : _m(m)
        , _data(m * n) {}

    iterator begin() { return {{_data.begin(), _m}, _m}; }
    iterator end() { return {{_data.end(), _m}, _m}; }

    const_iterator begin() const { return {{_data.begin(), _m}, _m}; }
    const_iterator end() const { return {{_data.end(), _m}, _m}; }

    std::uint8_t* data() { return _data.data(); }
    std::uint8_t const* data() const { return _data.data(); }

    std::size_t size() const { return _data.size(); }

private:
    unsigned _m;
    storage _data;
};
