#pragma once

#include "json.h"
#include "memory.h"
#include "view.h"
#include "dataset.h"
#include <vector>

using value_type = std::uint8_t;
using vec_view = view<std::vector<value_type>::const_iterator>;

struct stream {
    virtual ~stream() = default;

    virtual vec_view next() = 0;

    std::size_t osize() const { return _osize; }

protected:
    stream(std::size_t osize)
        : _osize(osize) {}

private:
    const std::size_t _osize;
};
