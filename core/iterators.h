#pragma once

#include <cassert>
#include <iterator>

namespace core {
namespace detail {
    template <class I>
    using StepIteratorBase = std::iterator<
            std::forward_iterator_tag, I,
            typename std::iterator_traits<I>::difference_type>;
}

template <class I> struct StepIterator : detail::StepIteratorBase<I> {
    using difference_type =
            typename std::iterator_traits<StepIterator>::difference_type;
    using reference = typename std::iterator_traits<StepIterator>::reference;

public:
    StepIterator() : _it(), _step(0) {}
    StepIterator(I iterator, difference_type step)
        : _it(iterator), _step(step) {}

    reference operator*() { return _it; }
    const reference operator*() const { return _it; }

    StepIterator& operator++() {
        _it += _step;
        return *this;
    }

    StepIterator operator++(int) {
        auto self = *this;
        _it += _step;
        return self;
    }

    bool operator==(StepIterator const& b) const {
        assert(_step == b._step);
        return _it == b._it;
    }
    bool operator!=(StepIterator const& b) const { return !(*this == b); }

private:
    I _it;
    difference_type _step;
};

template <class I, class Difference = typename StepIterator<I>::difference_type>
StepIterator<I> make_step_iterator(I iterator, Difference step) {
    return {iterator, step};
}
} // namespace core
