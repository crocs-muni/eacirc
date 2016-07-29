#pragma once

#include <stdexcept>
#include <type_traits>

namespace core {

struct bad_maybe_access : std::logic_error {
    bad_maybe_access()
        : logic_error("Maybe does not contains a value!") {}
};

template <class T> struct maybe {
    using value_type = T;

    maybe()
        : _set(false) {}

    maybe(value_type &&value)
        : _set(true) {
        new (&_storage) value_type(std::move(value));
    }

    maybe(const value_type &value)
        : _set(true) {
        new (&_storage) value_type(value);
    }

    maybe(maybe &&o)
        : _set(o.set) {
        if (_set)
            _construct(std::move(o));
    }

    maybe(const maybe &o)
        : _set(o.set) {
        if (_set)
            _construct(o);
    }

    ~maybe() {
        if (_set)
            _destruct();
    }

    maybe &operator=(value_type &&value) {
        if (_set)
            unsafe_value() = std::move(value);
        else {
            new (&_storage) value_type(std::move(value));
            _set = true;
        }
        return *this;
    }

    maybe &operator=(const value_type &value) {
        if (_set)
            unsafe_value() = value;
        else {
            new (&_storage) value_type(value);
            _set = true;
        }
        return *this;
    }

    maybe &operator=(maybe &&o) {
        if (_set && o._set)
            unsafe_value() = std::move(o.unsafe_value());
        else if (_set)
            _destruct();
        else if (o._set)
            _construct(std::move(o));
        _set = o._set;
        return *this;
    }

    maybe &operator=(const maybe &o) {
        if (_set && o._set)
            unsafe_value() = o.unsafe_value();
        else if (_set)
            _destruct();
        else if (o._set)
            _construct(o);
        _set = o._set;
        return *this;
    }

    void swap(maybe &o) {
        using std::swap;

        if (_set && o._set)
            swap(unsafe_value(), o.unsafe_value());
        else if (_set) {
            o._construct(std::move(o));
            _destruct();
        } else if (o._set) {
            _construct(std::move(o));
            o._destruct();
        }
        swap(_set, o._set);
    }

    void reset() {
        if (_set)
            _destruct();
        _set = false;
    }

    template <class... Args> void emplace(Args... args) {
        if (_set)
            _destruct();
        new (&_storage) value_type(std::forward<Args>(args)...);
        _set = true;
    }

    value_type &value() & {
        if (!_set)
            throw bad_maybe_access{};
        return unsafe_value();
    }

    const value_type &value() const & {
        if (!_set)
            throw bad_maybe_access{};
        return unsafe_value();
    }

    value_type &&value() && {
        if (!_set)
            throw bad_maybe_access{};
        return std::move(unsafe_value());
    }

    const value_type &&value() const && {
        if (!_set)
            throw bad_maybe_access{};
        return std::move(unsafe_value());
    }

    template <class U> value_type value_or(value_type &&default_value) const & {
        return bool(*this) ? **this
                           : static_cast<T>(std::forward<U>(default_value));
    }

    template <class U> value_type value_or(value_type &&default_value) && {
        bool(*this) ? std::move(**this)
                    : static_cast<T>(std::forward<U>(default_value));
    }

    bool has_value() const { return _set; }

    value_type *operator->() { return &value(); }
    const value_type *operator->() const { return &value(); }

    value_type &operator*() & { return value(); }
    value_type &&operator*() && { return value(); }

    const value_type &operator*() const & { return value(); }
    const value_type &&operator*() const && { return value(); }

    operator bool() const { return has_value(); }

private:
    bool _set;
    std::aligned_storage_t<sizeof(value_type), alignof(value_type)> _storage;

    value_type &unsafe_value() {
        return reinterpret_cast<value_type &>(_storage);
    }

    const value_type &unsafe_value() const {
        return reinterpret_cast<const value_type &>(_storage);
    }

    void _destruct() { unsafe_value().~value_type(); }

    void _construct(maybe &&o) {
        new (&_storage) value_type(std::move(o.unsafe_value()));
    }

    void _construct(const maybe &o) {
        new (&_storage) value_type(o.unsafe_value());
    }
};

template <class T> void swap(maybe<T> &a, maybe<T> &b) { a.swap(b); }

} // namespace core
