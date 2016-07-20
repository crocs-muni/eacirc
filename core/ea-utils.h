#pragma once

#include <memory>

namespace ea {

template <class T> struct factory {
    virtual ~factory() = default;

    virtual std::unique_ptr<T> create() = 0;
};

} // namespace ea
