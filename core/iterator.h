#pragma once

#include "base.h"
#include "range.h"
#include <iterator>

namespace detail {
template <class I>
struct step_iterator
        : std::iterator<
                  std::forward_iterator_tag, typename std::iterator_traits<I>::value_type,
                  typename std::iterator_traits<I>::difference_type,
                  typename std::iterator_traits<I>::pointer,
                  typename std::iterator_traits<I>::reference> {};
} // namespace detail

template <class I> struct step_iterator : detail::step_iterator<I>, Eq {
    using pointer = typename detail::step_iterator<I>::pointer;
    using reference = typename detail::step_iterator<I>::reference;
    using difference_type = typename detail::step_iterator<I>::difference_type;

private:
    I _it;
    difference_type _n;

public:
    step_iterator() : _it(), _n(0) {}
    step_iterator(I iterator, difference_type n) : _it(iterator), _n(n) {}

    reference operator*() const { return *_it; }
    pointer operator->() const { return &*_it; }

    step_iterator& operator++() {
        _it += _n;
        return *this;
    }

    step_iterator operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

    bool operator==(step_iterator const& b) const { return _it == b._it; }
};

template <class I>
step_iterator<I> make_step_iterator(I iterator, typename step_iterator<I>::difference_type n) {
    return {iterator, n};
}

namespace detail {
template <class I>
struct range_iterator : std::iterator<
                                typename std::iterator_traits<I>::iterator_category, const Range<I>,
                                typename std::iterator_traits<I>::difference_type> {};
} // namespace detail

template <class I> struct range_iterator : detail::range_iterator<I>, Eq {
    using iterator_category = typename detail::range_iterator<I>::iterator_category;
    using pointer = typename detail::range_iterator<I>::pointer;
    using reference = typename detail::range_iterator<I>::reference;
    using difference_type = typename detail::range_iterator<I>::difference_type;

private:
    Range<I> _range;

public:
    range_iterator() = default;
    range_iterator(Range<I> range) : _range(range) {}
    range_iterator(I beg, I end) : _range(beg, end) {}
    range_iterator(I beg, difference_type n) : _range(beg, n) {}

    reference operator*() const { return _range; }
    pointer operator->() const { return &_range; }

    range_iterator& operator++() {
        ++(_range._beg);
        ++(_range._end);
        return *this;
    }

    range_iterator& operator--() {
        --(_range._beg);
        --(_range._end);
        return *this;
    }

    range_iterator operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

    range_iterator operator--(int) {
        auto self = *this;
        --(*this);
        return self;
    }

    range_iterator& operator+=(difference_type n) {
        _range._beg += n;
        _range._end += n;
        return *this;
    }

    bool operator==(range_iterator const& b) const { return _range == b._range; }
};

template <class I> range_iterator<I> make_range_iterator(Range<I> range) {
    return {range};
}

template <class I> range_iterator<I> make_range_iterator(I beg, I end) {
    return {beg, end};
}

template <class I>
range_iterator<I> make_range_iterator(I beg, typename range_iterator<I>::difference_type n) {
    return {beg, n};
}

namespace detail {
template <class T> struct discard_iterator : std::iterator<std::input_iterator_tag, T> {};
} // namespace detail

template <class T> struct discard_iterator : detail::discard_iterator<T>, Eq {
    using reference = typename detail::discard_iterator<T>::reference;

private:
    T _dummy;

public:
    discard_iterator() = default;

    reference operator*() { return _dummy; }

    discard_iterator& operator++() { return *this; }
    discard_iterator operator++(int) { return *this; }

    bool operator==(discard_iterator const&) const { return true; }
};
