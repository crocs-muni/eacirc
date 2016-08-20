#pragma once

#include <cstddef>

struct max_iterations {
    max_iterations(std::size_t target)
        : _counter(0)
        , _target(target) {
    }

    template <typename T> bool operator()(T const&) {
        return _counter != _target;
    }

    max_iterations& operator++() {
        ++_counter;
        return *this;
    }

    max_iterations operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

private:
    std::size_t _counter;
    std::size_t _target;
};
