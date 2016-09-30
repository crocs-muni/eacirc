#pragma once

#include "view.h"
#include <cstdint>
#include <limits>
#include <stdexcept>

using byte = unsigned char;

using byte_view = view<byte*>;
using const_byte_view = view<const byte*>;

struct stream_error : std::runtime_error {
    stream_error(std::string const& what)
        : runtime_error(what) {}

    stream_error(char const* what)
        : runtime_error(what) {}
};

struct stream {
    virtual ~stream() = default;

    virtual std::size_t output_block_size() const { return 0u; }

    virtual void read(byte_view out) = 0;
};

struct counter {
    using value_type = std::uint64_t;
    using reference = value_type const&;
    using pointer = value_type const*;
    using difference_type = std::int64_t;
    using iterator_tag = std::forward_iterator_tag;

    counter()
        : _data(0ul) {}

    reference operator*() const { return _data; }
    pointer operator->() const { return &_data; }

    std::size_t size() const { return sizeof(_data); }
    std::uint8_t const* data() const { return reinterpret_cast<std::uint8_t const*>(&_data); }

    counter& operator++() {
        ++_data;
        return *this;
    }

    counter operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

    bool operator==(counter const& rhs) const { return _data == rhs._data; }
    bool operator!=(counter const& rhs) const { return !(*this == rhs); }

private:
    value_type _data;
};
