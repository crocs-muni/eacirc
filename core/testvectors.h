#pragma once

#include "base.h"
#include "iterators.h"
#include "range.h"
#include <vector>

template <class Storage = std::vector<u8>> struct Basic_TestVectors {
    using iterator = BlockIterator<typename Storage::iterator>;
    using const_iterator = BlockIterator<typename Storage::const_iterator>;
    using difference_type = typename Storage::difference_type;
    using size_type = typename Storage::size_type;

private:
    difference_type _tv_size;
    Storage _data;

public:
    Basic_TestVectors(size_t tv_size, size_t num_of_tvs)
        : _tv_size(tv_size), _data(_tv_size * num_of_tvs) {}

    void swap(Basic_TestVectors& other) {
        using std::swap;
        swap(_data, other._data);
        swap(_data, other._data);
    }

    size_type size() const { _data.size() / _tv_size; }

    iterator begin() { return {_data.begin(), _tv_size}; }
    const_iterator begin() const { return {_data.begin(), _tv_size}; }

    iterator end() { return {_data.end(), _tv_size}; }
    const_iterator end() const { return {_data.end(), _tv_size}; }
};

template <class T> void swap(Basic_TestVectors<T>& a, Basic_TestVectors<T>& b) {
    a.swap(b);
}

using TestVectors = Basic_TestVectors<std::vector<u8>>;
