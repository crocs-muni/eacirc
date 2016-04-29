#pragma once

#include "base.h"
#include "iterator.h"
#include <vector>

struct Dataset : Swappable {
    using Storage = std::vector<u8>;
    using Vec = Range<typename Storage::iterator>;

    using iterator = step_iterator<range_iterator<Storage::iterator>>;
    using const_iterator = step_iterator<range_iterator<Storage::const_iterator>>;
    using difference_type = typename Storage::difference_type;

private:
    unsigned _tv_size;
    unsigned _num_of_tvs;
    Storage _data;

public:
    Dataset() : Dataset(0, 0) {}
    Dataset(unsigned tv_size, unsigned num_of_tvs)
        : _tv_size(tv_size), _num_of_tvs(num_of_tvs), _data(tv_size * num_of_tvs) {}

    Dataset(Dataset&& b) : Dataset() { swap(b); }
    Dataset(Dataset const& b) : _tv_size(b._tv_size), _num_of_tvs(b._num_of_tvs), _data(b._data) {}

    Dataset& operator=(Dataset b) {
        swap(b);
        return *this;
    }

    void swap(Dataset& b) {
        using std::swap;
        swap(_tv_size, b._tv_size);
        swap(_num_of_tvs, b._num_of_tvs);
        swap(_data, b._data);
    }

    iterator begin() {
        return make_step_iterator(make_range_iterator(_data.begin(), _tv_size), _tv_size);
    }
    const_iterator begin() const {
        return make_step_iterator(make_range_iterator(_data.begin(), _tv_size), _tv_size);
    }

    iterator end() {
        return make_step_iterator(make_range_iterator(_data.end(), _tv_size), _tv_size);
    }
    const_iterator end() const {
        return make_step_iterator(make_range_iterator(_data.end(), _tv_size), _tv_size);
    }

    u8* data() { return _data.data(); }
    u8 const* data() const { return _data.data(); }

    unsigned tv_size() const { return _tv_size; }
    size_t num_of_tvs() const { return _num_of_tvs; }
};
