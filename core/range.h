#pragma once

#include <iterator>

template <class I> struct Range {
    using value_type = typename std::iterator_traits<I>::value_type;

private:
    I _beg;
    I _end;

public:
    constexpr Range(I beg, I end) : _beg(beg), _end(end) {}

    constexpr I begin() const { return _beg; }
    constexpr I end() const { return _end; }
};

template <class T>
constexpr auto make_range(T& container) -> Range<typename T::iterator> {
    return {container.begin(), container.end()};
}

template <class T>
constexpr auto make_range(T const& container)
        -> Range<typename T::const_iterator> {
    return {container.begin(), container.end()};
}
