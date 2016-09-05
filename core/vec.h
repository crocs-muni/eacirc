#pragma once

#include <cstdint>

template <std::size_t Size> struct vec {
    using value_type = std::uint8_t;

    using reference = value_type&;
    using const_reference = value_type const&;

    using pointer = value_type*;
    using const_pointer = value_type const*;

    using iterator = pointer;
    using const_iterator = const_pointer;

    iterator begin() { return _data; }
    iterator end() { return _data + Size; }

    const_iterator begin() const { return _data; }
    const_iterator end() const { return _data + Size; }

    reference operator[](unsigned i) { return _data[i]; }
    const_reference operator[](unsigned i) const { return _data[i]; }

    pointer data() { return _data; }
    const_pointer data() const { return _data; }

    constexpr std::size_t size() const { return Size; }

private:
    value_type _data[Size];
};
