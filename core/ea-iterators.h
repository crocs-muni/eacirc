#pragma once

#include <iterator>

namespace ea {

/**
 * @brief simple holder for 2 iterators
 */
template <class I> struct range {
    template <class> friend struct range_iterator;

    range()
        : _beg()
        , _end() {}

    range(I beg, I end)
        : _beg(beg)
        , _end(end) {}

    I begin() const { return _beg; }
    I end() const { return _end; }

    bool operator==(const range &b) const noexcept {
        return _beg == b._beg && _end == b._end;
    }

    bool operator!=(const range &b) const noexcept { return !((*this) == b); }

private:
    I _beg;
    I _end;
};

/**
 * @brief convenient utility to make range from iterator and distance
 */
template <class I, class S = typename std::iterator_traits<I>::difference_type>
auto make_range(I beg, S n) -> range<I> {
    return {beg, std::next(beg, n)};
}

/**
 * @brief convenientutility to make rake from an container
 */
template <class T>
auto make_range(T &&container) -> range<decltype(std::begin(container))> {
    return {std::begin(container), std::end(container)};
}

namespace _impl {

template <class I> struct range_iterator {
    typedef std::iterator<typename std::iterator_traits<I>::iterator_category,
                          const range<I>,
                          typename std::iterator_traits<I>::difference_type>
            type;
};

} // namespace _impl

/**
 * @brief iterator over \a range that advances both iterators that are held by
 * \a range
 */
template <class I> struct range_iterator : _impl::range_iterator<I>::type {
    using base = typename _impl::range_iterator<I>::type;

    range_iterator() = default;

    range_iterator(range<I> range)
        : _range(range) {}

    typename base::reference operator*() const { return _range; }
    typename base::pointer operator->() const { return &_range; }

    range_iterator &operator++() {
        ++(_range._beg);
        ++(_range._end);
        return *this;
    }

    range_iterator operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

    bool operator==(const range_iterator &b) const {
        return _range == b._range;
    }

    bool operator!=(const range_iterator &b) const { return !(*this == b); }

private:
    range<I> _range;
};

/**
 * @brief convenient utility to make an \a range_iterator from \a range
 */
template <class I>
auto make_range_iterator(range<I> range) -> range_iterator<I> {
    return {range};
}

/**
 * @brief convenient utility to make an \a range_iterator from two iterators
 */
template <class I>
auto make_range_iterator(I begin, I end) -> range_iterator<I> {
    return {{begin, end}};
}

/**
 * @brief convenient utility to make an \a range_iterator from iterator and
 * distance
 */
template <class I, class S = typename std::iterator_traits<I>::difference_type>
auto make_range_iterator(I iterator, S n) -> range_iterator<I> {
    return {make_range(iterator, n)};
}

namespace _impl {

template <class I> struct step_iterator {
    typedef std::iterator<std::forward_iterator_tag,
                          typename std::iterator_traits<I>::value_type,
                          typename std::iterator_traits<I>::difference_type,
                          typename std::iterator_traits<I>::pointer,
                          typename std::iterator_traits<I>::reference>
            type;
};

} // namespace _impl

/**
 * @brief an iterator adaptor that in one incrementation it advances an
 * underlying iterator several times
 */
template <class I> struct step_iterator : _impl::step_iterator<I>::type {
    using base = typename _impl::step_iterator<I>::type;
    using difference_type = typename base::difference_type;

    step_iterator() = default;

    step_iterator(I iterator, difference_type step)
        : _iterator(iterator)
        , _step(step) {}

    typename base::reference operator*() const { return *_iterator; }
    typename base::pointer operator->() const { return &*_iterator; }

    step_iterator &operator++() {
        std::advance(_iterator, _step);
        return *this;
    }

    step_iterator operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

    bool operator==(const step_iterator &b) const;
    bool operator!=(const step_iterator &b) const { return !(*this == b); }

private:
    I _iterator;
    difference_type _step;
};

/**
 * @brief covenient utility to create \a step_iterator from iterator and a step
 * length
 */
template <class I, class S = typename step_iterator<I>::difference_type>
auto make_step_iterator(I iterator, S step) -> step_iterator<I> {
    return {iterator, step};
}

namespace _impl {

template <class T> struct sequence_iterator {
    using type = std::iterator<std::forward_iterator_tag, T,
                               std::make_signed_t<T>, const T *, const T &>;
};

} // namespace _impl

/**
 * @brief iterator that generates a sequence of numbers on the fly
 */
template <class T>
struct sequence_iterator : _impl::sequence_iterator<T>::type {
    using base = typename _impl::sequence_iterator<T>::type;
    using value_type = typename base::value_type;

    constexpr sequence_iterator()
        : _value(0) {}

    constexpr explicit sequence_iterator(value_type value)
        : _value(value) {}

    constexpr typename base::reference operator*() const { return _value; }

    sequence_iterator &operator++() {
        ++_value;
        return *this;
    }

    sequence_iterator operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

    constexpr bool operator==(const sequence_iterator b) const {
        return _value == b._value;
    }

    constexpr bool operator!=(const sequence_iterator b) const {
        return !(*this == b);
    }

private:
    value_type _value;
};

/**
 * @brief utility for creating a sequence_iterator from a range [from, to)
 */
template <class T>
constexpr range<sequence_iterator<T>> sequence(T from, T to) {
    return {sequence_iterator<T>{from}, sequence_iterator<T>{to}};
}

/**
 * @brief utility for creating a sequence_iterator from a range [0, to)
 */
template <class T> constexpr range<sequence_iterator<T>> sequence(T to) {
    return sequence<T>(0, to);
}

} // namespace ea
