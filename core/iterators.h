#pragma once

#include "view.h"

namespace _impl {

    template <typename Iterator, typename T = void>
    using enable_if_forward_t = typename std::enable_if<
            std::is_base_of<std::forward_iterator_tag,
                            typename std::iterator_traits<Iterator>::iterator_category>::value,
            T>::type;

    template <typename Iterator, typename T = void>
    using enable_if_bidirectional_t = typename std::enable_if<
            std::is_base_of<std::bidirectional_iterator_tag,
                            typename std::iterator_traits<Iterator>::iterator_category>::value,
            T>::type;

    template <typename Iterator, typename T = void>
    using enable_if_random_access_t = typename std::enable_if<
            std::is_base_of<std::random_access_iterator_tag,
                            typename std::iterator_traits<Iterator>::iterator_category>::value,
            T>::type;

} // namespace _impl

template <typename Iterator> struct step_iterator {
    using pointer = typename std::iterator_traits<Iterator>::pointer;
    using reference = typename std::iterator_traits<Iterator>::reference;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;

    template <typename = _impl::enable_if_forward_t<Iterator>>
    step_iterator()
        : _it()
        , _n(0) {}

    step_iterator(Iterator it, difference_type n)
        : _it(it)
        , _n(n) {}

    step_iterator& operator++() {
        std::advance(_it, _n);
        return *this;
    }

    step_iterator operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

    reference operator*() const { return *_it; }
    pointer operator->() const { return &*_it; }

    bool operator==(step_iterator const& rhs) const { return std::distance(_it, rhs._it) < _n; }
    bool operator!=(step_iterator const& rhs) const { return !(*this == rhs); }

    auto operator--() -> _impl::enable_if_bidirectional_t<Iterator, step_iterator&> {
        std::advance(_it, -_n);
        return *this;
    }

    auto operator--(int) -> _impl::enable_if_bidirectional_t<Iterator, step_iterator> {
        auto self = *this;
        --(*this);
        return self;
    }

    auto operator+=(difference_type n)
            -> _impl::enable_if_random_access_t<Iterator, step_iterator&> {
        _it += n * _n;
        return *this;
    }

    auto operator-=(difference_type n)
            -> _impl::enable_if_random_access_t<Iterator, step_iterator&> {
        return *this += -n;
    }

    auto operator+(difference_type n) const
            -> _impl::enable_if_random_access_t<Iterator, step_iterator> {
        auto temp = *this;
        return temp += n;
    }

    auto operator-(difference_type n) const
            -> _impl::enable_if_random_access_t<Iterator, step_iterator> {
        auto temp = *this;
        return temp -= n;
    }

    auto operator[](difference_type n) -> _impl::enable_if_random_access_t<Iterator, reference> {
        return *(*this + n);
    }

    auto operator-(step_iterator const& rhs) const
            -> _impl::enable_if_random_access_t<Iterator, difference_type> {
        ASSERT((_it - rhs._it) % _n == 0);
        return (_it - rhs._it) / _n;
    }

    auto operator<(step_iterator const& rhs) const
            -> _impl::enable_if_random_access_t<Iterator, bool> {
        ASSERT(_n == rhs._n);
        return _it < rhs._it;
    }

    auto operator>(step_iterator const& rhs) const
            -> _impl::enable_if_random_access_t<Iterator, bool> {
        ASSERT(_n == rhs._n);
        return _it > rhs._it;
    }

    auto operator>=(step_iterator const& rhs) const
            -> _impl::enable_if_random_access_t<Iterator, bool> {
        return !(this < rhs);
    }

    auto operator<=(step_iterator const& rhs) const
            -> _impl::enable_if_random_access_t<Iterator, bool> {
        return !(*this > rhs);
    }

    friend auto operator+(difference_type lhs, step_iterator const& rhs)
            -> _impl::enable_if_random_access_t<Iterator, step_iterator> {
        return rhs + lhs;
    }

private:
    Iterator _it;
    difference_type _n;
};

template <typename Iterator> struct view_iterator {
    using value_type = view<Iterator>;
    using pointer = value_type const*;
    using reference = value_type const&;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;

    template <typename = _impl::enable_if_forward_t<Iterator>>
    view_iterator()
        : _view() {}

    view_iterator(Iterator beg, Iterator end)
        : view_iterator(value_type(beg, end)) {}

    view_iterator(Iterator it, difference_type n)
        : view_iterator(value_type(it, n)) {}

    view_iterator(value_type view)
        : _view(view) {}

    view_iterator& operator++() {
        ++(_view._beg);
        ++(_view._end);
        return *this;
    }

    view_iterator operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

    reference operator*() const { return _view; }
    pointer operator->() const { return &_view; }

    bool operator==(view_iterator const& rhs) const { return _view._end == rhs._view._end; }
    bool operator!=(view_iterator const& rhs) const { return !(*this == rhs); }

    auto operator--() -> _impl::enable_if_bidirectional_t<Iterator, view_iterator&> {
        --(_view._beg);
        --(_view._end);
        return *this;
    }

    auto operator--(int) -> _impl::enable_if_bidirectional_t<Iterator, view_iterator> {
        auto self = *this;
        --(*this);
        return self;
    }

    auto operator+=(difference_type n)
            -> _impl::enable_if_random_access_t<Iterator, view_iterator&> {
        _view._beg += n;
        _view._end += n;
        return *this;
    }

    auto operator-=(difference_type n)
            -> _impl::enable_if_random_access_t<Iterator, view_iterator&> {
        return *this += -n;
    }

    auto operator+(difference_type n) const
            -> _impl::enable_if_random_access_t<Iterator, view_iterator> {
        auto temp = *this;
        return temp += n;
    }

    auto operator-(difference_type n) const
            -> _impl::enable_if_random_access_t<Iterator, view_iterator> {
        auto temp = *this;
        return temp -= n;
    }

    auto operator[](difference_type n) -> _impl::enable_if_random_access_t<Iterator, reference> {
        return *(*this + n);
    }

    auto operator-(view_iterator const& rhs) const
            -> _impl::enable_if_random_access_t<Iterator, difference_type> {
        ASSERT(_view._beg - rhs._view._beg == _view._end - rhs._view._end);
        return _view._beg - rhs._view._beg;
    }

    auto operator<(view_iterator const& rhs) const
            -> _impl::enable_if_random_access_t<Iterator, bool> {
        ASSERT(_view._end < rhs._view._end);
        return _view._beg < rhs._view._beg;
    }

    auto operator>(view_iterator const& rhs) const
            -> _impl::enable_if_random_access_t<Iterator, bool> {
        ASSERT(_view._end > rhs._view._end);
        return _view._beg > rhs._view._beg;
    }

    auto operator>=(view_iterator const& rhs) const
            -> _impl::enable_if_random_access_t<Iterator, bool> {
        return !(this < rhs);
    }

    auto operator<=(view_iterator const& rhs) const
            -> _impl::enable_if_random_access_t<Iterator, bool> {
        return !(*this > rhs);
    }

    friend auto operator+(difference_type lhs, view_iterator const& rhs)
            -> _impl::enable_if_random_access_t<Iterator, view_iterator> {
        return rhs + lhs;
    }

private:
    value_type _view;
};
