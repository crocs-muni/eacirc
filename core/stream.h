#pragma once

#include <cstdint>

namespace core {

struct stream {
    virtual ~stream() = default;
    virtual std::size_t read(std::uint8_t *data, std::size_t size) = 0;
};

} // namespace core
