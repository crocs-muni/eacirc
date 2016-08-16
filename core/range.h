#pragma once

#include "debug.h"
#include <iterator>

namespace core {

/**
 * @brief an iterator interval [beg, end) wich is acting like container
 */
template <class I> struct range {
    using difference_type = typename std::iterator_traits<I>::difference_type;
    using value_type = typename std::iterator_traits<I>::value_type;
    using reference = typename std::iterator_traits<I>::reference;
    using pointer = typename std::iterator_traits<I>::pointer;
    using iterator_category = typename std::iterator_traits<I>::iterator_category;

    range()
        : _beg()
        , _end() {
    }

    range(I beg, I end)
        : _beg(beg)
        , _end(end) {
    }

    range(I beg, difference_type size)
        : range(beg, beg + size) {
    }

    I begin() const {
        return _beg;
    }
    I end() const {
        return _end;
    }

    range<I> take(difference_type n) const {
        ASSERT(n <= std::distance(_beg, _end));
        return {_beg, std::next(_beg, n)};
    }

    range<I> drop(difference_type n) const {
        ASSERT(n <= std::distance(_beg, _end));
        return {std::next(_beg, n), _end};
    }

    bool operator==(const range &b) const noexcept {
        return _beg == b._beg && _end == b._end;
    }

    bool operator!=(const range &b) const noexcept {
        return !((*this) == b);
    }

private:
    I _beg;
    I _end;
};

/**
 * @brief convinient function for creating \a range<I> from two iterators
 */
template <class I> range<I> make_range(I beg, I end) {
    return {beg, end};
}

/**
 * @brief convinient function for creating \a range<I> from and iterator and a
 * size
 */
template <class I, class S> range<I> make_range(I beg, S size) {
    return {beg, static_cast<typename range<I>::difference_type>(size)};
}

} // namespace core
