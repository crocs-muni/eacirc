#pragma once

#include <cstdint>
#include <vector>

template <unsigned Size> struct vec {
    using value_type = std::uint8_t;

    using reference = value_type&;
    using const_reference = const value_type&;

    using iterator = value_type*;
    using const_iterator = const value_type*;

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

template <unsigned vec_size> struct dataset {
    using storage = std::vector<vec<vec_size>>;

    using value_type = typename storage::value_type;

    using iterator = typename storage::iterator;
    using const_iterator = typename storage::const_iterator;

    dataset(std::size_t size)
        : _vectors(size) {
    }

    void clear() {
        _vectors.clear();
    }

    void push_back(value_type&& value) {
        _vectors.push_back(std::move(value));
    }

    iterator begin() {
        return _vectors.begin();
    }

    const_iterator begin() const {
        return _vectors.begin();
    }

    iterator end() {
        return _vectors.end();
    }

    const_iterator end() const {
        return _vectors.begin();
    }

    std::uint8_t* data() {
        return reinterpret_cast<std::uint8_t*>(_vectors.data());
    }

    const std::uint8_t* data() const {
        return reinterpret_cast<const std::uint8_t*>(_vectors.data());
    }

    std::size_t size() const {
        return _vectors.size();
    }

private:
    storage _vectors;
};
