#pragma once

#include <cstdint>
#include <vector>

template <unsigned Size> struct vec {
    using value_type = std::uint8_t;

    using reference = value_type &;
    using const_reference = const value_type &;

    using iterator = value_type *;
    using const_iterator = const value_type *;

    vec() = default;

    template <class I>
    vec(I beg, I end)
        : vec() {
        std::copy(beg, end, _data);
    }

    template <class I, class S = std::size_t>
    vec(I beg, S size)
        : vec(beg, beg + size) {
    }

    reference operator[](unsigned i) {
        return _data[i];
    }

    const_reference operator[](unsigned i) const {
        return _data[i];
    }

    iterator begin() {
        return _data;
    }

    iterator end() {
        return _data + Size;
    }

    const_iterator end() const {
        return _data + Size;
    }

    const_iterator begin() const {
        return _data;
    }

private:
    value_type _data[Size];
};

template <unsigned Datavec> struct dataset {
    using value_type = vec<Datavec>;

    using storage = std::vector<value_type>;
    using iterator = typename storage::iterator;
    using const_iterator = typename storage::const_iterator;

    void swap(dataset &o) {
        using std::swap;
        swap(_set, o._set);
    }

    friend void swap(const dataset &a, const dataset &b) {
        a.swap(b);
    }

    iterator begin() {
        return _set.begin();
    }

    const_iterator begin() const {
        return _set.begin();
    }

    iterator end() {
        return _set.end();
    }

    const_iterator end() const {
        return _set.end();
    }

private:
    std::vector<value_type> _set;
};
