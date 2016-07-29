#pragma once

#include "maybe.h"
#include "utils.h"

namespace core {

/**
 * @brief an iterator interval [beg, end) wich is acting like container
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
 * @brief an iterator that handles a pack of iterators as a tupple
 */
template <class... I> struct zip {
    zip(I &&... its)
        : _its(std::forward<I>(its)...) {}

    auto operator*() const {
        return map(_its, [](auto x) { return *x; });
    }

    zip &operator++() {
        map(_its, [](auto x) { return ++x; });
        return *this;
    }

    zip operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

    bool operator==(const zip &b) const {
        return std::get<0>(_its) == std::get<0>(b._its);
    }

    bool operator!=(const zip &b) const { return !(*this == b); }

private:
    std::tuple<I...> _its;
};

/**
 * @brief an iterator that lazilly applies a functor to a group of dereferenced
 * iterators and returns the result
 */
template <class F, class... I> struct zip_with {
    using value_type
            = std::result_of_t<F(decltype(*std::declval<zip<I...>>()))>;

    zip_with(F function, I... iterators)
        : _iterator(std::move(iterators)...)
        , _function(std::move(function)) {}

    auto operator*() const {
        if (_value)
            _value = apply(_function, *_iterator);
        return *_value;
    }

    zip_with &operator++() {
        ++_iterator;
        return *this;
    }

    zip_with operator++(int) {
        auto self = *this;
        ++(*this);
        return self;
    }

    bool operator==(const zip_with &b) const {
        return _iterator == b._iterator;
    }

    bool operator!=(const zip_with &b) const { return !(*this == b); }

private:
    zip<I...> _iterator;
    maybe<value_type> _value;
    F _function;
};

template <class I> struct jumper {
    auto operator*() const { return *_iterator; }

    jumper &operator++();

private:
    I _iterator;
};

} // namespace core
