#pragma once

#include "dataset.h"
#include <algorithm>
#include <limits>
#include <vector>

struct counter {
    using value_type = std::uint8_t;
    using pointer = typename std::add_pointer<value_type>::type;
    using const_pointer = typename std::add_pointer<const value_type>::type;
    using iterator = std::vector<value_type>::iterator;
    using const_iterator = std::vector<value_type>::const_iterator;

    counter(std::size_t size)
        : _data(size) {
        std::fill(_data.begin(), _data.end(), std::numeric_limits<value_type>::min());
    }

    void increment() {
        for (value_type& value : _data) {
            if (value != std::numeric_limits<value_type>::max()) {
                ++value;
                break;
            }
            value = std::numeric_limits<value_type>::min();
        }
    }

    iterator begin() { return _data.begin(); }
    const_iterator begin() const { return _data.begin(); }

    iterator end() { return _data.end(); }
    const_iterator end() const { return _data.end(); }

    pointer data() { return _data.data(); }
    const_pointer data() const { return _data.data(); }

    std::size_t size() const { return _data.size(); }

private:
    std::vector<value_type> _data;
};

struct stream {
    virtual ~stream() = default;

    virtual void read(dataset& set) = 0;
};
