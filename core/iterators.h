#pragma once

#include "range.h"
#include <iterator>

template <class I>
struct BlockIterator : std::iterator<
                               std::forward_iterator_tag, Range<I>, std::ptrdiff_t, Range<I> const*,
                               Range<I> const&> {
public:
    using Size = typename std::iterator_traits<I>::difference_type;

    BlockIterator() : _it(), _size(0) {}
    BlockIterator(I iterator, Size block_size) : _it(iterator), _size(block_size) {}

    Range<I> operator*() const { return {_it, _it + _size}; }

    BlockIterator& operator++() {
        _it += _size;
        return *this;
    }

    BlockIterator operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

    bool operator==(BlockIterator const& b) const {
        assert(_size == b._size);
        return _it <= b._it && b._it < _it + _size;
    }
    bool operator!=(BlockIterator const& b) const { return !(*this == b); }

private:
    I _it;
    Size _size;
};
