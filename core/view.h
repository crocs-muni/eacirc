#pragma once

#include "debug.h"
#include <iterator>

/**
 * @brief an iterator interval [beg, end) wich is acting as container, but it does not owns any data
 */
template <typename Iterator> struct view {
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using reference = typename std::iterator_traits<Iterator>::reference;
    using pointer = typename std::iterator_traits<Iterator>::pointer;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;

    view()
        : _beg()
        , _end() {}

    view(Iterator beg, Iterator end)
        : _beg(beg)
        , _end(end) {}

    view(Iterator beg, difference_type n)
        : view(beg, beg + n) {}

    Iterator begin() const { return _beg; }
    Iterator end() const { return _end; }

    std::size_t size() const { return std::size_t(std::distance(_beg, _end)); }

    view take(difference_type n) const {
        ASSERT(n <= std::distance(_beg, _end));
        return {_beg, std::next(_beg, n)};
    }

    view drop(difference_type n) const {
        ASSERT(n <= std::distance(_beg, _end));
        return {std::next(_beg, n), _end};
    }

private:
    Iterator _beg;
    Iterator _end;
};

/**
 * @brief convinient function for creating \a view<I> from a pair od iterators
 */
template <typename I> auto make_view(I beg, I end) -> view<I> {
    return {beg, end};
}

/**
 * @brief convinient function for creating \a view<I> from an iterator and a size
 */
template <typename I, typename S> auto make_view(I beg, S n) -> view<I> {
    return {beg, typename view<I>::difference_type(n)};
}
