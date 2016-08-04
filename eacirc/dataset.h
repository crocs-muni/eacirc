#pragma once

#include <cstdint>
#include <vector>

template <unsigned Size> struct datavec {
    using value_type = std::uint8_t;

    using storage = value_type[Size];
    using iterator = value_type *;
    using const_iterator = const value_type *;

    datavec() = default;

    template <class I>
    datavec(I beg, I end)
        : datavec() {
        std::copy(beg, end, _data);
    }

    template <class I, class S>
    datavec(I beg, S size)
        : datavec(beg, beg + size) {}

    iterator begin() { return _data; }
    const_iterator begin() const { return _data; }

    iterator end() { return _data + Size; }
    const_iterator end() const { return _data + Size; }

private:
    storage _data;
};

template <unsigned Datavec> struct dataset {
    using value_type = datavec<Datavec>;

    using storage = std::vector<value_type>;
    using iterator = typename storage::iterator;
    using const_iterator = typename storage::const_iterator;

    void swap(dataset &o) {
        using std::swap;
        swap(_set, o._set);
    }

    iterator begin() { return _set.begin(); }
    const_iterator begin() const { return _set.begin(); }

    iterator end() { return _set.end(); }
    const_iterator end() const { return _set.end(); }

private:
    std::vector<value_type> _set;
};

template <std::size_t S> void swap(dataset<S> &a, dataset<S> &b) { a.swap(b); }
