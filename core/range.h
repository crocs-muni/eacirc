#pragma once

#include "base.h"
#include <iterator>

template <class I> struct Range : Eq {
    using iterator = I;
    using difference_type = typename std::iterator_traits<I>::difference_type;

    template <class> friend struct range_iterator;

private:
    iterator _beg;
    iterator _end;

public:
    Range() : _beg(), _end() {}
    Range(iterator beg, iterator end) : _beg(beg), _end(end) {}
    Range(iterator beg, difference_type n) : Range(beg, beg + n) {}

    iterator begin() const { return _beg; }
    iterator end() const { return _end; }

    size_t size() const { return std::distance(_beg, _end); }

    bool operator==(Range const& b) const { return _beg == b._beg && _end == b._end; }
};

template <class I> Range<I> make_range(I beg, I end) {
    return {beg, end};
}

template <class I> Range<I> make_range(I iterator, typename Range<I>::difference_type n) {
    return {iterator, iterator + n};
}
